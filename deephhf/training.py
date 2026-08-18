from __future__ import annotations

from pathlib import Path
import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from .config import Config
from .data import WindowDataset, RecordingDataset
from .models import ECGEncoder, RecordingTransformer
from .utils import device


def _pos_weight(labels: list[float]) -> torch.Tensor:
    positive = sum(labels)
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        raise ValueError("Training data must contain both outcome classes")
    return torch.tensor(negative / positive, dtype=torch.float32)


def _loader(dataset, batch_size: int, shuffle: bool, workers: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def embed_recordings(encoder: ECGEncoder, ecg: torch.Tensor, window_samples: int, step_samples: int, batch_windows: int = 64) -> torch.Tensor:
    """Embed equally spaced windows for a batch of fixed recordings."""
    batch, total = ecg.shape
    starts = torch.arange(0, total - window_samples + 1, step_samples, device=ecg.device)
    all_embeddings = []
    for sample in ecg:
        windows = torch.stack([sample[s : s + window_samples] for s in starts]).unsqueeze(1)
        parts = [encoder.encode(piece) for piece in windows.split(batch_windows)]
        all_embeddings.append(torch.cat(parts))
    return torch.stack(all_embeddings)


def _run_epoch_encoder(model, loader, loss_fn, optimizer, scaler, use_amp, train: bool) -> tuple[float, np.ndarray, np.ndarray]:
    model.train(train)
    losses, predictions, labels = [], [], []
    target_device = device()
    for ecg, label in loader:
        ecg, label = ecg.to(target_device, non_blocking=True), label.to(target_device)
        with torch.autocast(device_type=target_device.type, enabled=use_amp):
            logits, _ = model(ecg)
            loss = loss_fn(logits, label)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        losses.append(loss.item() * len(label))
        predictions.extend(torch.sigmoid(logits).detach().cpu().numpy())
        labels.extend(label.cpu().numpy())
    return sum(losses) / len(loader.dataset), np.asarray(labels), np.asarray(predictions)


def train_encoder(manifest: str | Path, config: Config, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    train_set = WindowDataset(manifest, config.data, "train")
    val_set = WindowDataset(manifest, config.data, "val")
    if not len(train_set) or not len(val_set):
        raise ValueError("Manifest requires non-empty train and val splits for encoder training")
    if len({record["label"] for record in val_set.records}) != 2:
        raise ValueError("Validation split must contain both outcome classes for AUROC early stopping")
    model = ECGEncoder(config.model.encoder_channels, config.model.embedding_dim, config.model.dropout).to(device())
    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight([r["label"] for r in train_set.records]).to(device()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.encoder_lr, weight_decay=config.training.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.training.amp and torch.cuda.is_available())
    train_loader = _loader(train_set, config.training.batch_size, True, config.training.num_workers)
    val_loader = _loader(val_set, config.training.batch_size, False, config.training.num_workers)
    best_auc, patience, best_state = -np.inf, 0, None
    history = []
    for epoch in range(1, config.training.epochs + 1):
        train_loss, _, _ = _run_epoch_encoder(model, train_loader, criterion, optimizer, scaler, scaler.is_enabled(), True)
        val_loss, y, p = _run_epoch_encoder(model, val_loader, criterion, optimizer, scaler, scaler.is_enabled(), False)
        auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auroc": auc})
        logging.getLogger("deephhf").info("Encoder epoch %d: train_loss=%.4f val_loss=%.4f val_auroc=%.4f", epoch, train_loss, val_loss, auc)
        if auc > best_auc:
            best_auc, patience, best_state = auc, 0, copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= config.training.patience: break
    checkpoint = output_dir / "encoder_best.pt"
    torch.save({"model": best_state, "config": config.to_dict(), "best_val_auroc": best_auc, "history": history}, checkpoint)
    return checkpoint


def _run_epoch_sequence(encoder, sequence, loader, loss_fn, optimizer, scaler, config, train: bool):
    encoder.eval(); sequence.train(train)
    losses, labels, predictions = [], [], []
    target_device = device()
    for ecg, label, _ in loader:
        ecg, label = ecg.to(target_device), label.to(target_device)
        embeddings = embed_recordings(encoder, ecg, config.data.samples_per_window, config.data.sequence_step_seconds * config.data.target_rate_hz)
        with torch.autocast(device_type=target_device.type, enabled=scaler.is_enabled()):
            logits = sequence(embeddings)
            loss = loss_fn(logits, label)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        losses.append(loss.item() * len(label)); labels.extend(label.cpu().numpy()); predictions.extend(torch.sigmoid(logits).detach().cpu().numpy())
    return sum(losses) / len(loader.dataset), np.asarray(labels), np.asarray(predictions)


def train_sequence(manifest: str | Path, encoder_checkpoint: str | Path, config: Config, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(encoder_checkpoint, map_location=device(), weights_only=False)
    encoder = ECGEncoder(config.model.encoder_channels, config.model.embedding_dim, config.model.dropout).to(device())
    encoder.load_state_dict(payload["model"]); encoder.requires_grad_(False); encoder.eval()
    sequence = RecordingTransformer(config.model.embedding_dim, config.model.transformer_heads, config.model.transformer_layers, config.model.transformer_ff_dim, config.model.dropout).to(device())
    train_set, val_set = RecordingDataset(manifest, "train"), RecordingDataset(manifest, "val")
    if not len(train_set) or not len(val_set): raise ValueError("Manifest requires non-empty train and val splits")
    if len({record["label"] for record in val_set.records}) != 2:
        raise ValueError("Validation split must contain both outcome classes for AUROC early stopping")
    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight([r["label"] for r in train_set.records]).to(device()))
    optimizer = torch.optim.AdamW(sequence.parameters(), lr=config.training.sequence_lr, weight_decay=config.training.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.training.amp and torch.cuda.is_available())
    best_auc, patience, best_state, history = -np.inf, 0, None, []
    for epoch in range(1, config.training.epochs + 1):
        tr_loss, _, _ = _run_epoch_sequence(encoder, sequence, _loader(train_set, config.training.sequence_batch_size, True, config.training.num_workers), criterion, optimizer, scaler, config, True)
        va_loss, y, p = _run_epoch_sequence(encoder, sequence, _loader(val_set, config.training.sequence_batch_size, False, config.training.num_workers), criterion, optimizer, scaler, config, False)
        auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_auroc": auc})
        logging.getLogger("deephhf").info("Sequence epoch %d: train_loss=%.4f val_loss=%.4f val_auroc=%.4f", epoch, tr_loss, va_loss, auc)
        if auc > best_auc: best_auc, patience, best_state = auc, 0, copy.deepcopy(sequence.state_dict())
        else:
            patience += 1
            if patience >= config.training.patience: break
    checkpoint = output_dir / "sequence_best.pt"
    torch.save({"model": best_state, "encoder_checkpoint": str(encoder_checkpoint), "config": config.to_dict(), "best_val_auroc": best_auc, "history": history}, checkpoint)
    return checkpoint
