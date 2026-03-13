import os
import argparse
import glob
import numpy as np
from collections import defaultdict
import json


def print_msg(msg):
    print(f"[ANALYZE_RAW] {msg}", flush=True)


def load_pcd_labels(pcd_path):
    """
    Wczytuje etykiety z surowego pliku PCD.
    
    Returns:
        np.ndarray: Tablica z etykietami punktów
    """
    print_msg(f"Parsing: {os.path.basename(pcd_path)}")
    
    fields_line = None
    data_start_idx = None

    with open(pcd_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line.startswith("FIELDS"):
                fields_line = line
            if line.startswith("DATA"):
                data_start_idx = i + 1
                break

    if fields_line is None or data_start_idx is None:
        raise RuntimeError(f"Invalid PCD header in file: {pcd_path}")

    parts = fields_line.split()
    fields = [f.lower() for f in parts[1:]]
    
    # Wczytaj dane
    data = np.loadtxt(pcd_path, skiprows=data_start_idx)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != len(fields):
        raise RuntimeError(
            f"File {pcd_path}: number of columns ({data.shape[1]}) "
            f"doesn't match FIELDS ({len(fields)})"
        )

    field_to_col = {name: idx for idx, name in enumerate(fields)}

    # Znajdź kolumnę z etykietami
    label_field_candidates = ["label", "class", "classification"]
    label_idx = None
    for name in label_field_candidates:
        if name in field_to_col:
            label_idx = field_to_col[name]
            break

    if label_idx is not None:
        labels = data[:, label_idx].astype(np.int32)
    else:
        print_msg(f"WARNING: {pcd_path}: missing label field – setting all labels to 0")
        labels = np.zeros((data.shape[0],), dtype=np.int32)

    return labels


def analyze_pcd_file(pcd_path):
    """
    Analizuje pojedynczy plik PCD i zwraca statystyki.
    
    Returns:
        dict: Statystyki zawierające liczbę punktów per klasa
    """
    try:
        labels = load_pcd_labels(pcd_path)
        
        # Policz wystąpienia każdej klasy
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        stats = {
            "total_points": int(labels.shape[0]),
            "classes": {}
        }
        
        for label, count in zip(unique_labels, counts):
            stats["classes"][int(label)] = int(count)
        
        return stats
    
    except Exception as e:
        print_msg(f"ERROR processing {pcd_path}: {e}")
        return None


def main():
    dataset_root = "/data/ZAHA/raw"
    output_file = "/data/ZAHA/zaha_statistics.json"


    if not os.path.isdir(dataset_root):
        print_msg(f"ERROR: Directory not found: {dataset_root}")
        return

    print_msg("Scanning raw dataset...")

    splits = ["training", "validation", "test"]
    all_stats = {
        "global": {
            "total_points": 0,
            "total_files": 0,
            "classes": defaultdict(int)
        },
        "splits": {}
    }

    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        
        if not os.path.isdir(split_dir):
            print_msg(f"WARNING: Split directory not found: {split_dir}")
            continue

        # Znajdź wszystkie pliki PCD
        pcd_files = sorted(glob.glob(os.path.join(split_dir, "*.pcd")))
        
        print_msg(f"Processing split '{split}': {len(pcd_files)} PCD files")

        split_stats = {
            "total_points": 0,
            "total_files": len(pcd_files),
            "classes": defaultdict(int),
            "files": {}
        }

        for pcd_file in pcd_files:
            file_name = os.path.basename(pcd_file)
            scene_name = os.path.splitext(file_name)[0]
            
            stats = analyze_pcd_file(pcd_file)
            
            if stats is None:
                continue
            
            # Aktualizuj statystyki dla splitu
            split_stats["total_points"] += stats["total_points"]
            split_stats["files"][scene_name] = stats
            
            for class_id, count in stats["classes"].items():
                split_stats["classes"][class_id] += count
                all_stats["global"]["classes"][class_id] += count
            
            # Aktualizuj globalne statystyki
            all_stats["global"]["total_points"] += stats["total_points"]
        
        # Konwertuj defaultdict na zwykły dict
        split_stats["classes"] = dict(split_stats["classes"])
        all_stats["splits"][split] = split_stats
        
        print_msg(f"  Split '{split}': {split_stats['total_points']:,} total points")

    # Konwertuj globalne defaultdict na dict
    all_stats["global"]["classes"] = dict(all_stats["global"]["classes"])
    all_stats["global"]["total_files"] = sum(
        split_data["total_files"] for split_data in all_stats["splits"].values()
    )

    # Zapisz statystyki do pliku JSON
    output_path = os.path.join(os.path.dirname(dataset_root), output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print_msg(f"Statistics saved to: {output_path}")
    print_msg("\n=== GLOBAL STATISTICS ===")
    print_msg(f"Total files: {all_stats['global']['total_files']}")
    print_msg(f"Total points: {all_stats['global']['total_points']:,}")
    print_msg("\nPoints per class:")
    
    for class_id in sorted(all_stats['global']['classes'].keys()):
        count = all_stats['global']['classes'][class_id]
        percentage = (count / all_stats['global']['total_points']) * 100
        print_msg(f"  Class {class_id}: {count:,} points ({percentage:.2f}%)")

    print_msg("\n=== SPLIT STATISTICS ===")
    for split in splits:
        if split in all_stats["splits"]:
            split_data = all_stats["splits"][split]
            print_msg(f"\n{split.upper()}:")
            print_msg(f"  Files: {split_data['total_files']}")
            print_msg(f"  Total points: {split_data['total_points']:,}")
            print_msg(f"  Points per class:")
            for class_id in sorted(split_data['classes'].keys()):
                count = split_data['classes'][class_id]
                percentage = (count / split_data['total_points']) * 100
                print_msg(f"    Class {class_id}: {count:,} ({percentage:.2f}%)")


main()