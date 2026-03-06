"""
Sequence Feature Extraction for HMM

Extracts time-domain and frequency-domain features from each window
while preserving sequence structure for HMM training.

Output: Each sequence becomes a feature matrix (n_windows × n_features)
"""

import csv
import json
import os
import numpy as np
from collections import defaultdict

INPUT_DIR = "sequence_data"
OUTPUT_DIR = "sequence_features"


def load_window_csv(csv_path):
    """Load a single window CSV file."""
    data = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"])
            })
    return data


def extract_time_domain_features(data):
    """
    Extract time-domain features from sensor data.
    
    Returns dict with features for x, y, z axes.
    """
    if not data:
        return {}
    
    # Convert to numpy arrays
    x = np.array([p["x"] for p in data])
    y = np.array([p["y"] for p in data])
    z = np.array([p["z"] for p in data])
    
    features = {}
    
    # Per-axis features
    for axis_name, axis_data in [("x", x), ("y", y), ("z", z)]:
        features[f"{axis_name}_mean"] = np.mean(axis_data)
        features[f"{axis_name}_std"] = np.std(axis_data)
        features[f"{axis_name}_var"] = np.var(axis_data)
        features[f"{axis_name}_min"] = np.min(axis_data)
        features[f"{axis_name}_max"] = np.max(axis_data)
        features[f"{axis_name}_range"] = np.ptp(axis_data)
    
    # Signal Magnitude Area (SMA)
    features["sma"] = np.sum(np.abs(x)) + np.sum(np.abs(y)) + np.sum(np.abs(z))
    features["sma"] /= len(x)
    
    # Correlation between axes
    features["corr_xy"] = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
    features["corr_xz"] = np.corrcoef(x, z)[0, 1] if len(x) > 1 else 0.0
    features["corr_yz"] = np.corrcoef(y, z)[0, 1] if len(x) > 1 else 0.0
    
    # Magnitude
    mag = np.sqrt(x**2 + y**2 + z**2)
    features["mag_mean"] = np.mean(mag)
    features["mag_std"] = np.std(mag)
    
    return features


def extract_frequency_domain_features(data, n_components=5):
    """
    Extract frequency-domain features using FFT.
    
    Returns dict with frequency features.
    """
    if not data or len(data) < 4:
        return {}
    
    x = np.array([p["x"] for p in data])
    y = np.array([p["y"] for p in data])
    z = np.array([p["z"] for p in data])
    
    features = {}
    
    for axis_name, axis_data in [("x", x), ("y", y), ("z", z)]:
        # Compute FFT
        fft_vals = np.fft.fft(axis_data)
        fft_mag = np.abs(fft_vals[:len(fft_vals)//2])  # Only positive frequencies
        
        if len(fft_mag) == 0:
            continue
        
        # Dominant frequency (index of max magnitude)
        features[f"{axis_name}_dom_freq"] = np.argmax(fft_mag)
        
        # Spectral energy
        features[f"{axis_name}_spectral_energy"] = np.sum(fft_mag**2)
        
        # Top N FFT components (magnitudes)
        n = min(n_components, len(fft_mag))
        top_indices = np.argsort(fft_mag)[-n:][::-1]
        for i, idx in enumerate(top_indices):
            features[f"{axis_name}_fft_{i}"] = fft_mag[idx]
    
    return features


def extract_window_features(accel_data, gyro_data):
    """
    Extract all features from accelerometer and gyroscope data for one window.
    
    Returns: feature vector as dict
    """
    features = {}
    
    # Time-domain features
    accel_time = extract_time_domain_features(accel_data)
    gyro_time = extract_time_domain_features(gyro_data)
    
    for key, val in accel_time.items():
        features[f"accel_{key}"] = val
    for key, val in gyro_time.items():
        features[f"gyro_{key}"] = val
    
    # Frequency-domain features
    accel_freq = extract_frequency_domain_features(accel_data)
    gyro_freq = extract_frequency_domain_features(gyro_data)
    
    for key, val in accel_freq.items():
        features[f"accel_{key}"] = val
    for key, val in gyro_freq.items():
        features[f"gyro_{key}"] = val
    
    # Handle NaN values
    for key in features:
        if np.isnan(features[key]) or np.isinf(features[key]):
            features[key] = 0.0
    
    return features


def process_sequence(sequence_dir, split, activity):
    """
    Process one sequence (all windows) and extract features.
    
    Returns:
        - sequence_id
        - feature matrix (n_windows × n_features)
        - feature names
    """
    # Load metadata
    metadata_path = os.path.join(sequence_dir, "sequence_metadata.json")
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    sequence_id = metadata["sequence_id"]
    n_windows = metadata["n_windows"]
    
    # Extract features for each window
    all_features = []
    feature_names = None
    
    for i in range(n_windows):
        window_id = f"w{i:02d}"
        
        # Load window data
        accel_path = os.path.join(sequence_dir, f"{window_id}_accel.csv")
        gyro_path = os.path.join(sequence_dir, f"{window_id}_gyro.csv")
        
        accel_data = load_window_csv(accel_path)
        gyro_data = load_window_csv(gyro_path)
        
        # Extract features
        features = extract_window_features(accel_data, gyro_data)
        
        # Store feature names (from first window)
        if feature_names is None:
            feature_names = sorted(features.keys())
        
        # Convert to array in consistent order
        feature_vector = [features.get(name, 0.0) for name in feature_names]
        all_features.append(feature_vector)
    
    # Convert to numpy array
    feature_matrix = np.array(all_features)  # Shape: (n_windows, n_features)
    
    return sequence_id, feature_matrix, feature_names, metadata


def save_sequence_features(sequence_id, feature_matrix, feature_names, metadata, split):
    """
    Save features for one sequence.
    """
    output_dir = os.path.join(OUTPUT_DIR, split, sequence_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "features.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["window_idx"] + feature_names)
        for i, row in enumerate(feature_matrix):
            writer.writerow([i] + list(row))
    
    # Save as numpy array
    npy_path = os.path.join(output_dir, "features.npy")
    np.save(npy_path, feature_matrix)
    
    # Save metadata with feature info
    metadata["n_features"] = len(feature_names)
    metadata["feature_names"] = feature_names
    metadata["feature_shape"] = feature_matrix.shape
    
    metadata_path = os.path.join(output_dir, "features_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return feature_matrix.shape


def main():
    """Process all sequences and extract features."""
    print("="*70)
    print("  SEQUENCE FEATURE EXTRACTION FOR HMM")
    print("="*70)
    print(f"\nInput directory: {INPUT_DIR}/")
    print(f"Output directory: {OUTPUT_DIR}/")
    
    # Clear output directory
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)
    
    # Load summary
    summary_path = os.path.join(INPUT_DIR, "summary.json")
    with open(summary_path) as f:
        summary = json.load(f)
    
    all_sequences_info = {"train": [], "test": []}
    stats = {"train": defaultdict(int), "test": defaultdict(int)}
    
    # Process each split
    for split in ["train", "test"]:
        print(f"\n{'─'*70}")
        print(f"Processing: {split.upper()} SET")
        print(f"{'─'*70}")
        
        sequences = summary[f"{split}_sequences"]
        
        for seq_meta in sequences:
            sequence_id = seq_meta["sequence_id"]
            activity = seq_meta["activity"]
            
            sequence_dir = os.path.join(INPUT_DIR, split, sequence_id)
            
            if not os.path.exists(sequence_dir):
                print(f"  ⚠ Sequence directory not found: {sequence_dir}")
                continue
            
            # Extract features
            seq_id, feature_matrix, feature_names, metadata = process_sequence(
                sequence_dir, split, activity
            )
            
            # Save features
            shape = save_sequence_features(seq_id, feature_matrix, feature_names, metadata, split)
            
            # Track statistics
            stats[split][activity] += 1
            all_sequences_info[split].append({
                "sequence_id": seq_id,
                "activity": activity,
                "n_windows": shape[0],
                "n_features": shape[1]
            })
            
            print(f"  ✓ {sequence_id:25s} → {shape[0]:2d} windows × {shape[1]:3d} features")
    
    # Save global feature summary
    feature_summary = {
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "train_sequences": all_sequences_info["train"],
        "test_sequences": all_sequences_info["test"],
        "statistics": dict(stats)
    }
    
    summary_path = os.path.join(OUTPUT_DIR, "feature_summary.json")
    with open(summary_path, "w") as f:
        json.dump(feature_summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("  FEATURE EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"\nFeature vector size: {len(feature_names)} features")
    print(f"\nFeature categories:")
    print(f"  - Time-domain: mean, std, var, min, max, range, SMA, correlations")
    print(f"  - Frequency-domain: dominant frequency, spectral energy, FFT components")
    print(f"\n{'Split':<10} {'Activity':<15} {'Sequences':>12} {'Total Windows':>15}")
    print(f"{'─'*70}")
    
    for split in ["train", "test"]:
        for activity in ["standing", "walking", "jumping", "still"]:
            n_seq = stats[split][activity]
            seqs = [s for s in all_sequences_info[split] if s["activity"] == activity]
            n_windows = sum(s["n_windows"] for s in seqs)
            print(f"{split:<10} {activity:<15} {n_seq:>12} {n_windows:>15}")
    
    total_train_seq = sum(stats["train"].values())
    total_test_seq = sum(stats["test"].values())
    total_train_win = sum(s["n_windows"] for s in all_sequences_info["train"])
    total_test_win = sum(s["n_windows"] for s in all_sequences_info["test"])
    
    print(f"{'─'*70}")
    print(f"{'TRAIN':<10} {'TOTAL':<15} {total_train_seq:>12} {total_train_win:>15}")
    print(f"{'TEST':<10} {'TOTAL':<15} {total_test_seq:>12} {total_test_win:>15}")
    print(f"{'─'*70}")


if __name__ == "__main__":
    main()
