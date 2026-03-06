"""
Sequence-Based Data Segmentation for HMM

This script segments sensor recordings into SEQUENTIAL time windows.
Each recording becomes a SEQUENCE of feature observations for HMM training.

- Each recording → sequence of N windows, each with features

Example: 10-second walking recording with 2-second windows
  → [window1, window2, window3, window4, window5] = 5 observations in sequence
"""

import csv
import os
import json
import glob

# Configuration
WINDOW_SIZE = 2.0  # 2 seconds per window
OVERLAP = 0.5      # 50% overlap between windows 
MIN_WINDOWS = 3    # Minimum windows per recording

# Split configuration
TRAIN_TEST_SPLIT = {
    "standing": {"train": [1,2,3,4,5,6,7,8], "test": [9,10,11,12]},
    "walking":  {"train": [1,2,3,4,5,6,7,8,9,10], "test": [11,12,13,14]},
    "jumping":  {"train": [1,2,3,4,5,6,7,8], "test": [9,10,11,12]},
    "still":    {"train": [1,2,3,4,5,6,7,8], "test": [9,10,11,12]}
}

OUTPUT_DIR = "sequence_data"


def read_sensor_data(csv_path, max_duration=None):
    """Read accelerometer or gyroscope CSV file."""
    data = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            elapsed = float(row["seconds_elapsed"])
            if max_duration and elapsed > max_duration:
                break
            data.append({
                "time": row["time"],
                "seconds_elapsed": elapsed,
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"])
            })
    return data


def create_sliding_windows(data, window_size, overlap):
    """
    Create sliding windows from sensor data.
    
    Args:
        data: List of sensor readings
        window_size: Window duration in seconds
        overlap: Overlap ratio (0.5 = 50% overlap)
    
    Returns:
        List of windows, each containing sensor readings
    """
    if not data:
        return []
    
    total_duration = data[-1]["seconds_elapsed"]
    stride = window_size * (1 - overlap)
    
    windows = []
    start_time = 0.0
    
    while start_time + window_size <= total_duration:
        end_time = start_time + window_size
        
        # Extract points in this window
        window_data = [
            point for point in data 
            if start_time <= point["seconds_elapsed"] < end_time
        ]
        
        if window_data:
            windows.append({
                "start": start_time,
                "end": end_time,
                "data": window_data
            })
        
        start_time += stride
    
    return windows


def save_window_csv(output_path, window_data):
    """Save a single window's data to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "seconds_elapsed", "x", "y", "z"])
        for point in window_data:
            writer.writerow([
                point["time"],
                f"{point['seconds_elapsed']:.6f}",
                f"{point['x']:.6f}",
                f"{point['y']:.6f}",
                f"{point['z']:.6f}"
            ])


def process_recording(folder_path, activity_name, recording_id, split):
    """
    Process one recording into a sequence of windows.
    
    Returns:
        dict with sequence metadata
    """
    accel_path = os.path.join(folder_path, "Accelerometer.csv")
    gyro_path = os.path.join(folder_path, "Gyroscope.csv")
    
    if not os.path.exists(accel_path) or not os.path.exists(gyro_path):
        print(f"  ⚠ Missing sensor files in {folder_path}")
        return None
    
    # Read sensor data
    accel_data = read_sensor_data(accel_path)
    gyro_data = read_sensor_data(gyro_path)
    
    if not accel_data or not gyro_data:
        print(f"  ⚠ Empty sensor data in {folder_path}")
        return None
    
    duration = accel_data[-1]["seconds_elapsed"]
    
    # Create sliding windows
    accel_windows = create_sliding_windows(accel_data, WINDOW_SIZE, OVERLAP)
    gyro_windows = create_sliding_windows(gyro_data, WINDOW_SIZE, OVERLAP)
    
    n_windows = min(len(accel_windows), len(gyro_windows))
    
    if n_windows < MIN_WINDOWS:
        print(f"  ⚠ Only {n_windows} windows in {folder_path} (min: {MIN_WINDOWS})")
        return None
    
    # Save windows
    sequence_id = f"{activity_name}_{recording_id:02d}"
    sequence_dir = os.path.join(OUTPUT_DIR, split, sequence_id)
    
    windows_metadata = []
    
    for i in range(n_windows):
        window_id = f"w{i:02d}"
        
        # Save accelerometer window
        accel_out = os.path.join(sequence_dir, f"{window_id}_accel.csv")
        save_window_csv(accel_out, accel_windows[i]["data"])
        
        # Save gyroscope window
        gyro_out = os.path.join(sequence_dir, f"{window_id}_gyro.csv")
        save_window_csv(gyro_out, gyro_windows[i]["data"])
        
        windows_metadata.append({
            "window_id": window_id,
            "start": accel_windows[i]["start"],
            "end": accel_windows[i]["end"],
            "accel_points": len(accel_windows[i]["data"]),
            "gyro_points": len(gyro_windows[i]["data"])
        })
    
    # Save sequence metadata
    metadata = {
        "sequence_id": sequence_id,
        "activity": activity_name,
        "recording_id": recording_id,
        "folder": os.path.basename(folder_path),
        "duration": duration,
        "n_windows": n_windows,
        "window_size": WINDOW_SIZE,
        "overlap": OVERLAP,
        "windows": windows_metadata
    }
    
    metadata_path = os.path.join(sequence_dir, "sequence_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def main():
    """Process all recordings into sequences."""
    print("="*70)
    print("  SEQUENCE-BASED DATA SEGMENTATION FOR HMM")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Window size: {WINDOW_SIZE}s")
    print(f"  Overlap: {OVERLAP*100:.0f}%")
    print(f"  Min windows per sequence: {MIN_WINDOWS}")
    print(f"  Output directory: {OUTPUT_DIR}/")
    
    # Clear output directory
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)
    
    all_sequences = {"train": [], "test": []}
    stats = {"train": {}, "test": {}}
    
    # Process each activity
    for activity in ["standing", "walking", "jumping", "still"]:
        print(f"\n{'─'*70}")
        print(f"Processing: {activity.upper()}")
        print(f"{'─'*70}")
        
        stats["train"][activity] = 0
        stats["test"][activity] = 0
        
        # Find all recordings for this activity
        pattern = f"{activity}*"
        folders = sorted(glob.glob(pattern))
        
        if not folders:
            print(f"  ⚠ No recordings found for {activity}")
            continue
        
        print(f"  Found {len(folders)} recordings")
        
        # Process each recording
        for folder in folders:
            # Get just the folder name (not full path)
            folder_name = os.path.basename(folder)
            
            # Extract recording number from folder name
            try:
                # Try pattern with underscore: "activity_N-date"
                if '_' in folder_name:
                    parts = folder_name.split('_')
                    if len(parts) > 1 and parts[1][0].isdigit():
                        rec_num = int(parts[1].split('-')[0])
                    else:
                        raise ValueError("Unexpected format after underscore")
                else:
                    # Try pattern without underscore: "activityN-date"
                    # Remove activity name and extract number
                    # For "still1-2026...", remove "still" → "1-2026..."
                    temp = folder_name.replace(activity, '')  # Remove activity name
                    rec_num = int(temp.split('-')[0])
            except Exception as e:
                print(f"  ⚠ Could not parse recording number from {folder_name}: {e}")
                continue
            
            # Determine split
            if rec_num in TRAIN_TEST_SPLIT[activity]["train"]:
                split = "train"
            elif rec_num in TRAIN_TEST_SPLIT[activity]["test"]:
                split = "test"
            else:
                print(f"  ⚠ Skipping {folder} (rec_num={rec_num} not in split config)")
                continue  # Skip if not in split config
            
            # Process recording
            metadata = process_recording(folder, activity, rec_num, split)
            
            if metadata:
                all_sequences[split].append(metadata)
                stats[split][activity] += 1
                print(f"  ✓ {folder} → {metadata['n_windows']} windows ({metadata['duration']:.1f}s) [{split}]")
    
    # Save global summary
    summary = {
        "window_size": WINDOW_SIZE,
        "overlap": OVERLAP,
        "min_windows": MIN_WINDOWS,
        "train_sequences": all_sequences["train"],
        "test_sequences": all_sequences["test"],
        "statistics": stats
    }
    
    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print("  SEGMENTATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n{'Split':<10} {'Activity':<15} {'Sequences':>12} {'Total Windows':>15}")
    print(f"{'─'*70}")
    
    for split in ["train", "test"]:
        for activity in ["standing", "walking", "jumping", "still"]:
            n_seq = stats[split][activity]
            seqs = [s for s in all_sequences[split] if s["activity"] == activity]
            n_windows = sum(s["n_windows"] for s in seqs)
            print(f"{split:<10} {activity:<15} {n_seq:>12} {n_windows:>15}")
    
    total_train_seq = sum(stats["train"].values())
    total_test_seq = sum(stats["test"].values())
    total_train_win = sum(s["n_windows"] for s in all_sequences["train"])
    total_test_win = sum(s["n_windows"] for s in all_sequences["test"])
    
    print(f"{'─'*70}")
    print(f"{'TRAIN':<10} {'TOTAL':<15} {total_train_seq:>12} {total_train_win:>15}")
    print(f"{'TEST':<10} {'TOTAL':<15} {total_test_seq:>12} {total_test_win:>15}")
    print(f"{'─'*70}")
    
    print(f"\nOutput structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── train/")
    print(f"  │   ├── <activity>_<id>/")
    print(f"  │   │   ├── w00_accel.csv")
    print(f"  │   │   ├── w00_gyro.csv")
    print(f"  │   │   ├── w01_accel.csv")
    print(f"  │   │   ├── w01_gyro.csv")
    print(f"  │   │   ├── ...")
    print(f"  │   │   └── sequence_metadata.json")
    print(f"  ├── test/")
    print(f"  │   └── (same structure)")
    print(f"  └── summary.json")


if __name__ == "__main__":
    main()
