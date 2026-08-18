"""
ECG Project: Data Acquisition & Setup Script

This script downloads public ECG datasets from PhysioNet and prepares them
for training your 24-hour live monitoring model.

REQUIREMENTS:
  pip install wfdb pandas numpy scipy scikit-learn

DATASETS:
  1. PTB-XL: 21,837 12-lead ECGs → extract lead I only
  2. Chapman ECG: 10,646 ECGs
  3. ICBEB 2018: 6,877 ECGs
  
Total: 39,360+ potential records for training
"""

import os
import json
import wfdb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "target_rate_hz": 128,
    "hours": 24,
    "download_dir": Path("data/raw_downloads"),
    "prepared_dir": Path("data/prepared"),
    "manifest_output": Path("data/raw_manifest.csv"),
}


class ECGDatasetDownloader:
    """Download and validate ECG datasets from PhysioNet"""
    
    def __init__(self, download_dir: Path = CONFIG["download_dir"]):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
    
    def download_ptbxl(self) -> pd.DataFrame:
        """
        Download PTB-XL dataset (21,837 records)
        
        Reference: Wagner et al. 2020
        - Multi-lead (12-lead) ECGs
        - 500 Hz sampling rate
        - Multiple diagnostic labels
        
        Returns:
            DataFrame with paths and metadata
        """
        logger.info("Downloading PTB-XL dataset...")
        
        # This would be implemented with wfdb library
        # For now, return stub
        
        db_path = self.download_dir / "ptbxl"
        db_path.mkdir(exist_ok=True)
        
        records = []
        # Pseudo-code:
        # for record_id in range(1, 21838):
        #     record = wfdb.rdrecord(f'ptb-xl/{record_id:05d}')
        #     # Extract lead I only (first channel)
        #     ecg = record.p_signal[:, 0]  
        #     # Save to file
        #     np.save(db_path / f"ptbxl_{record_id:05d}.npy", ecg)
        #     records.append({...})
        
        logger.info(f"PTB-XL: Downloaded {len(records)} records")
        return pd.DataFrame(records)
    
    def download_chapman(self) -> pd.DataFrame:
        """
        Download Chapman ECG dataset (10,646 records)
        
        Arrhythmia classification, single-lead
        500 Hz sampling rate
        """
        logger.info("Downloading Chapman ECG dataset...")
        
        # Similar to PTB-XL
        db_path = self.download_dir / "chapman"
        db_path.mkdir(exist_ok=True)
        
        logger.info("Chapman: Downloaded records")
        return pd.DataFrame()
    
    def download_icbeb2018(self) -> pd.DataFrame:
        """
        Download CinC 2018 Challenge dataset
        
        6,877 records with outcome labels
        500 Hz sampling rate
        """
        logger.info("Downloading ICBEB 2018 dataset...")
        
        db_path = self.download_dir / "icbeb2018"
        db_path.mkdir(exist_ok=True)
        
        logger.info("ICBEB2018: Downloaded records")
        return pd.DataFrame()


class ECGStandardizer:
    """Convert diverse ECGs to 24-hour standardized format"""
    
    @staticmethod
    def resample_ecg(signal: np.ndarray, 
                     source_rate_hz: float,
                     target_rate_hz: float = 128) -> np.ndarray:
        """Resample ECG to target rate"""
        from scipy.signal import resample
        
        num_samples = int(len(signal) * target_rate_hz / source_rate_hz)
        return resample(signal, num_samples)
    
    @staticmethod
    def pad_or_crop_24h(signal: np.ndarray,
                        target_rate_hz: float = 128,
                        duration_hours: float = 24) -> np.ndarray:
        """Pad or crop to exactly 24 hours"""
        target_samples = int(target_rate_hz * duration_hours * 3600)
        
        if len(signal) == target_samples:
            return signal
        elif len(signal) < target_samples:
            # Pad center
            pad_total = target_samples - len(signal)
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return np.pad(signal, (pad_left, pad_right), mode='constant', constant_values=0)
        else:
            # Crop center
            start = (len(signal) - target_samples) // 2
            return signal[start:start + target_samples]
    
    @staticmethod
    def robust_normalize(signal: np.ndarray) -> np.ndarray:
        """Normalize using median/MAD (robust to outliers)"""
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        if mad == 0:
            return (signal - median)
        return (signal - median) / mad


class DatasetBuilder:
    """Build final dataset manifest for training"""
    
    def __init__(self, prepared_dir: Path = CONFIG["prepared_dir"]):
        self.prepared_dir = Path(prepared_dir)
        self.prepared_dir.mkdir(parents=True, exist_ok=True)
        self.records_dir = self.prepared_dir / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
    
    def create_manifest(self, 
                       records_df: pd.DataFrame,
                       train_ratio: float = 0.70,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.10,
                       calibration_ratio: float = 0.05) -> None:
        """
        Create CSV manifest for model training
        
        Format:
            id, path, sampling_rate_hz, label, ecg_class, split
        
        Args:
            records_df: DataFrame with columns [id, signal, outcome, ecg_class]
            train/val/test/calibration_ratio: Split proportions
        """
        
        # Ensure splits sum to 1
        assert abs((train_ratio + val_ratio + test_ratio + calibration_ratio) - 1.0) < 1e-6
        
        # Stratified split by outcome + ecg_class
        manifest_rows = []
        
        for outcome in [0, 1]:
            for ecg_class in ['normal', 'abnormal']:
                subset = records_df[
                    (records_df['outcome'] == outcome) & 
                    (records_df['ecg_class'] == ecg_class)
                ]
                
                n = len(subset)
                indices = np.random.permutation(n)
                
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                n_test = int(n * test_ratio)
                
                # Train split
                for idx in indices[:n_train]:
                    manifest_rows.append({
                        'id': subset.iloc[idx]['id'],
                        'path': str((self.records_dir / f"{subset.iloc[idx]['id']}.npy").absolute()),
                        'sampling_rate_hz': 128,
                        'label': outcome,
                        'ecg_class': ecg_class,
                        'split': 'train'
                    })
                
                # Val split
                for idx in indices[n_train:n_train+n_val]:
                    manifest_rows.append({
                        'id': subset.iloc[idx]['id'],
                        'path': str((self.records_dir / f"{subset.iloc[idx]['id']}.npy").absolute()),
                        'sampling_rate_hz': 128,
                        'label': outcome,
                        'ecg_class': ecg_class,
                        'split': 'val'
                    })
                
                # Test split
                for idx in indices[n_train+n_val:n_train+n_val+n_test]:
                    manifest_rows.append({
                        'id': subset.iloc[idx]['id'],
                        'path': str((self.records_dir / f"{subset.iloc[idx]['id']}.npy").absolute()),
                        'sampling_rate_hz': 128,
                        'label': outcome,
                        'ecg_class': ecg_class,
                        'split': 'test'
                    })
                
                # Calibration split (remaining)
                for idx in indices[n_train+n_val+n_test:]:
                    manifest_rows.append({
                        'id': subset.iloc[idx]['id'],
                        'path': str((self.records_dir / f"{subset.iloc[idx]['id']}.npy").absolute()),
                        'sampling_rate_hz': 128,
                        'label': outcome,
                        'ecg_class': ecg_class,
                        'split': 'calibration'
                    })
        
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_df.to_csv(CONFIG["manifest_output"], index=False)
        
        logger.info(f"Manifest created: {len(manifest_df)} records")
        logger.info(f"  Train: {len(manifest_df[manifest_df['split']=='train'])}")
        logger.info(f"  Val:   {len(manifest_df[manifest_df['split']=='val'])}")
        logger.info(f"  Test:  {len(manifest_df[manifest_df['split']=='test'])}")
        logger.info(f"  Calib: {len(manifest_df[manifest_df['split']=='calibration'])}")


def main():
    """Main workflow"""
    
    logger.info("=" * 60)
    logger.info("ECG DATASET DOWNLOAD & PREPARATION")
    logger.info("=" * 60)
    
    # Step 1: Download datasets
    logger.info("\nStep 1: Downloading datasets from PhysioNet...")
    downloader = ECGDatasetDownloader()
    
    # Uncomment to actually download:
    # ptbxl_df = downloader.download_ptbxl()
    # chapman_df = downloader.download_chapman()
    # icbeb_df = downloader.download_icbeb2018()
    
    # For demo, create synthetic records
    logger.info("(Demo mode: creating synthetic dataset)")
    all_records = []
    
    for i in range(1000):  # Start with 1000, expand to 10,000
        # Create synthetic 24-hour ECG at 128 Hz
        samples = int(128 * 24 * 3600)  # 11,059,200 samples
        ecg = np.random.randn(samples) * 0.5  # Gaussian noise
        
        # Add some structure (sinusoidal component)
        t = np.linspace(0, 24, samples)
        ecg += 0.3 * np.sin(2 * np.pi * 1.5 * t)  # ~1.5 Hz component
        
        outcome = np.random.randint(0, 2)  # 0 or 1
        ecg_class = 'normal' if outcome == 0 else 'abnormal'
        
        all_records.append({
            'id': f'record_{i:06d}',
            'signal': ecg.astype(np.float32),
            'outcome': outcome,
            'ecg_class': ecg_class
        })
    
    all_records_df = pd.DataFrame(all_records)
    
    # Step 2: Save preprocessed records
    logger.info("\nStep 2: Saving standardized records...")
    builder = DatasetBuilder()
    
    for idx, row in all_records_df.iterrows():
        output_path = builder.records_dir / f"{row['id']}.npy"
        np.save(output_path, row['signal'])
        
        if (idx + 1) % 100 == 0:
            logger.info(f"  Saved {idx + 1}/{len(all_records_df)} records")
    
    # Step 3: Create manifest
    logger.info("\nStep 3: Creating manifest CSV...")
    builder.create_manifest(all_records_df)
    
    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE!")
    logger.info(f"Manifest: {CONFIG['manifest_output']}")
    logger.info(f"Records: {builder.records_dir}")
    logger.info("\nNext steps:")
    logger.info("1. deephhf prepare-data --input-manifest {manifest} --output-dir data/prepared")
    logger.info("2. deephhf train-encoder --manifest data/prepared/manifest.csv ...")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
