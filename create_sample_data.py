import numpy as np
import pandas as pd
from pathlib import Path

# Create sample ECG data directory
ecg_data_dir = Path("data/raw_ecg")
ecg_data_dir.mkdir(parents=True, exist_ok=True)

# Generate synthetic ECG data for demonstration
np.random.seed(42)
rows = []

for split in ["train", "val", "calibration", "test"]:
    for label in [0, 1]:
        for i in range(2):  # 2 subjects per split/label combo
            # Generate synthetic 24-hour ECG at 250 Hz
            # 24 hours * 3600 sec * 250 Hz = 21,600,000 samples
            duration_sec = 86400  # 24 hours
            sampling_rate = 250
            n_samples = duration_sec * sampling_rate
            
            # Simple synthetic signal: sine wave with noise
            t = np.arange(n_samples) / sampling_rate
            signal = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz base frequency (slow cardiac-like)
            signal += 0.1 * label * np.sin(2 * np.pi * 2 * t)  # HF cases have extra component
            signal += 0.05 * np.random.randn(n_samples)  # Add noise
            
            # Save as .npy file
            filename = f"{split}_label{label}_subject{i:02d}.npy"
            filepath = ecg_data_dir / filename
            np.save(filepath, signal.astype(np.float32))
            
            rows.append({
                "id": filepath.stem,
                "path": str(filepath.resolve()),
                "sampling_rate_hz": sampling_rate,
                "label": label,
                "ecg_class": "normal" if label == 0 else "abnormal",
                "split": split
            })

# Create manifest CSV
manifest_df = pd.DataFrame(rows)
manifest_path = Path("data/raw_manifest.csv")
manifest_df.to_csv(manifest_path, index=False)

print(f"Created {len(rows)} synthetic ECG records")
print(f"Manifest saved to: {manifest_path}")
print("\nFirst few rows:")
print(manifest_df.head())
