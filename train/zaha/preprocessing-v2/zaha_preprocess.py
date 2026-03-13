#!/usr/bin/env python3
# zaha_preprocess.py
"""
Główny skrypt preprocessingu datasetu ZAHA dla Pointcept/PTv3.

Użycie:
    python zaha_preprocess.py \
        --dataset_root /path/to/raw/zaha \
        --output_root /path/to/preprocessed/zaha \
        --num_workers 4

Struktura wejściowa (dataset_root):
    training/
        scene1.pcd
        scene2.pcd
    validation/
        ...
    test/
        ...

Struktura wyjściowa (output_root):
    train/
        scene1/
            coord.npy       (N, 3) float32
            segment.npy     (N,) int32
            color.npy       (N, 3) uint8
            normal.npy      (N, 3) float32
            height.npy      (N, 1) float32
            verticality.npy (N, 1) float32
            planarity.npy   (N, 1) float32
            linearity.npy   (N, 1) float32
            sphericity.npy  (N, 1) float32
        scene2/
            ...
    validation/
        ...
    test/
        ...
"""
import os
import argparse
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from zaha_pcd_io import load_pcd_with_labels
from zaha_feature_normals import compute_normals
from zaha_feature_height import compute_height_feature
from zaha_feature_geometry import compute_geometric_features


# ===============================
#  KONFIGURACJA CECH
# ===============================
USE_COLORS = False         # kolory z PCD
USE_NORMALS = True         # normalne powierzchni
USE_HEIGHT = True          # znormalizowana wysokość [0, 1]
USE_GEOMETRY = True        # cechy geometryczne (verticality, planarity, linearity, sphericity)

# Parametry
NORMAL_RADIUS = 0.3
NORMAL_MAX_NN = 30
GEOMETRY_K_NEIGHBORS = 30
GEOMETRY_RADIUS = 0.3

# Mapowanie nazw splitów (raw -> output)
SPLIT_MAPPING = {
    "training": "train",
    "validation": "validation",
    "test": "test",
}


def print_header(msg: str) -> None:
    print(f"[ZAHA] {msg}", flush=True)


def print_detail(msg: str) -> None:
    print(f"  {msg}", flush=True)


def parse_scene(
    split_raw: str,
    scene_name: str,
    dataset_root: str,
    output_root: str
) -> dict:
    """
    Przetwarza pojedynczą scenę.
    
    Returns:
        dict ze statystykami sceny
    """
    # Ścieżka źródłowa
    src_pcd = os.path.join(dataset_root, split_raw, scene_name + ".pcd")
    if not os.path.isfile(src_pcd):
        raise FileNotFoundError(f"Brak pliku: {src_pcd}")
    
    # Ścieżka docelowa
    split_out = SPLIT_MAPPING.get(split_raw, split_raw)
    save_dir = os.path.join(output_root, split_out, scene_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print_header(f"Processing: {split_raw}/{scene_name}")
    
    # === Wczytaj dane ===
    coords, colors_raw, labels = load_pcd_with_labels(src_pcd)
    num_points = coords.shape[0]
    print_detail(f"Loaded {num_points:,} points")
    
    # === Walidacja etykiet ===
    unique_labels = np.unique(labels)
    print_detail(f"Labels: {unique_labels.tolist()}")
    
    # === Segment (etykiety klas) ===
    # Kształt (N,) - Pointcept oczekuje 1D array
    segment = labels.astype(np.int32)
    
    # === Kolory ===
    if USE_COLORS and colors_raw is not None:
        colors = colors_raw
        print_detail("Using colors from PCD")
    else:
        colors = np.zeros((num_points, 3), dtype=np.uint8)
    
    # === Normalne ===
    normals = None
    if USE_NORMALS:
        normals = compute_normals(
            coords,
            radius=NORMAL_RADIUS,
            max_nn=NORMAL_MAX_NN,
        )
    
    # === Wysokość ===
    height = None
    if USE_HEIGHT:
        height = compute_height_feature(coords, mode="z_norm01")
        print_detail(f"Height range: [{height.min():.3f}, {height.max():.3f}]")
    
    # === Cechy geometryczne ===
    geom_features = None
    if USE_GEOMETRY:
        geom_features = compute_geometric_features(
            coords,
            normals=normals,
            k_neighbors=GEOMETRY_K_NEIGHBORS,
            radius=GEOMETRY_RADIUS,
            verbose=True
        )
    
    # ===============================
    #  ZAPIS
    # ===============================
    print_detail("Saving files...")
    
    # coord: (N, 3) float32
    np.save(os.path.join(save_dir, "coord.npy"), coords.astype(np.float32))
    
    # segment: (N,) int32
    np.save(os.path.join(save_dir, "segment.npy"), segment.astype(np.int32))
    
    # color: (N, 3) uint8
    np.save(os.path.join(save_dir, "color.npy"), colors.astype(np.uint8))
    
    # normal: (N, 3) float32
    if USE_NORMALS and normals is not None:
        np.save(os.path.join(save_dir, "normal.npy"), normals.astype(np.float32))
    
    # height: (N, 1) float32
    if USE_HEIGHT and height is not None:
        np.save(os.path.join(save_dir, "height.npy"), height.astype(np.float32))
    
    # Cechy geometryczne: każda (N, 1) float32
    if USE_GEOMETRY and geom_features is not None:
        for feat_name, feat_array in geom_features.items():
            np.save(
                os.path.join(save_dir, f"{feat_name}.npy"),
                feat_array.astype(np.float32)
            )
    
    print_header(f"Finished: {split_raw}/{scene_name}\n")
    
    # Statystyki
    unique_labels, counts = np.unique(labels, return_counts=True)
    return {
        "scene": scene_name,
        "split": split_raw,
        "num_points": num_points,
        "labels": dict(zip(unique_labels.tolist(), counts.tolist()))
    }


def verify_output(output_root: str) -> None:
    """Weryfikuje poprawność wygenerowanych plików."""
    print("\n" + "=" * 60)
    print(" WERYFIKACJA")
    print("=" * 60)
    
    expected_files = ["coord.npy", "segment.npy", "color.npy"]
    if USE_NORMALS:
        expected_files.append("normal.npy")
    if USE_HEIGHT:
        expected_files.append("height.npy")
    if USE_GEOMETRY:
        expected_files.extend(["verticality.npy", "planarity.npy", "linearity.npy", "sphericity.npy"])
    
    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(output_root, split)
        if not os.path.isdir(split_dir):
            continue
        
        scenes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        if not scenes:
            continue
        
        scene_dir = os.path.join(split_dir, scenes[0])
        print(f"\nSprawdzam: {split}/{scenes[0]}")
        
        n_points = None
        for fname in expected_files:
            fpath = os.path.join(scene_dir, fname)
            if os.path.exists(fpath):
                arr = np.load(fpath)
                print(f"  {fname}: shape={arr.shape}, dtype={arr.dtype}")
                
                if fname == "coord.npy":
                    n_points = arr.shape[0]
                elif n_points is not None and arr.shape[0] != n_points:
                    print(f"    BŁĄD: oczekiwano {n_points} punktów!")
            else:
                print(f"  {fname}: BRAK!")
        
        # Sprawdź orientację normalnych
        if USE_NORMALS:
            coord_path = os.path.join(scene_dir, "coord.npy")
            normal_path = os.path.join(scene_dir, "normal.npy")
            if os.path.exists(coord_path) and os.path.exists(normal_path):
                coords = np.load(coord_path)
                normals = np.load(normal_path)
                centroid = coords.mean(axis=0)
                outward = coords - centroid
                dot = np.sum(normals * outward, axis=1)
                n_outward = (dot >= 0).sum()
                print(f"\n  Orientacja normalnych:")
                print(f"    Na zewnątrz: {n_outward:,} / {len(dot):,} ({100*n_outward/len(dot):.1f}%)")
        
        break


def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing ZAHA dataset dla Pointcept/PTv3"
    )
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Ścieżka do surowego datasetu ZAHA"
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Ścieżka wyjściowa dla preprocessowanych danych"
    )
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="Liczba równoległych procesów (1 = sekwencyjnie)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Weryfikuj wygenerowane pliki po zakończeniu"
    )
    args = parser.parse_args()
    
    dataset_root = args.dataset_root
    output_root = args.output_root
    num_workers = args.num_workers
    
    os.makedirs(output_root, exist_ok=True)
    
    # === Header ===
    print("=" * 60)
    print(" ZAHA Preprocessing")
    print("=" * 60)
    print(f"Input:   {dataset_root}")
    print(f"Output:  {output_root}")
    print(f"Workers: {num_workers}")
    print()
    print("Features:")
    print(f"  Colors:   {USE_COLORS}")
    print(f"  Normals:  {USE_NORMALS} (radius={NORMAL_RADIUS}, max_nn={NORMAL_MAX_NN})")
    print(f"  Height:   {USE_HEIGHT}")
    print(f"  Geometry: {USE_GEOMETRY} (k={GEOMETRY_K_NEIGHBORS}, radius={GEOMETRY_RADIUS})")
    print()
    
    # === Skanuj strukturę ===
    scene_list = []
    for split_raw in SPLIT_MAPPING.keys():
        split_dir = os.path.join(dataset_root, split_raw)
        if not os.path.isdir(split_dir):
            print(f"[WARNING] Brak katalogu: {split_dir}")
            continue
        
        pcd_files = sorted(glob.glob(os.path.join(split_dir, "*.pcd")))
        print(f"Split '{split_raw}': {len(pcd_files)} plików")
        
        for p in pcd_files:
            scene_name = os.path.splitext(os.path.basename(p))[0]
            scene_list.append((split_raw, scene_name))
    
    total_scenes = len(scene_list)
    if total_scenes == 0:
        print("[ERROR] Nie znaleziono żadnych plików .pcd!")
        return
    
    print(f"\nŁącznie: {total_scenes} scen")
    print("=" * 60 + "\n")
    
    # === Przetwarzanie ===
    all_stats = []
    
    if num_workers <= 1:
        for i, (split_raw, scene_name) in enumerate(scene_list):
            try:
                stats = parse_scene(split_raw, scene_name, dataset_root, output_root)
                all_stats.append(stats)
            except Exception as e:
                print(f"[ERROR] {split_raw}/{scene_name}: {e}")
            print(f"Progress: {i+1}/{total_scenes} ({100*(i+1)/total_scenes:.1f}%)\n")
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {}
            for split_raw, scene_name in scene_list:
                fut = pool.submit(parse_scene, split_raw, scene_name, dataset_root, output_root)
                futures[fut] = (split_raw, scene_name)
            
            done = 0
            for fut in as_completed(futures):
                done += 1
                split_raw, scene_name = futures[fut]
                try:
                    stats = fut.result()
                    all_stats.append(stats)
                except Exception as e:
                    print(f"[ERROR] {split_raw}/{scene_name}: {e}")
                print(f"Progress: {done}/{total_scenes} ({100*done/total_scenes:.1f}%)")
    
    # === Weryfikacja ===
    if args.verify:
        verify_output(output_root)
    
    # === Podsumowanie ===
    print("\n" + "=" * 60)
    print(" PREPROCESSING COMPLETED")
    print("=" * 60)
    print(f"Output: {output_root}")
    print(f"Scenes: {len(all_stats)}/{total_scenes}")
    
    if all_stats:
        total_points = sum(s["num_points"] for s in all_stats)
        print(f"Total points: {total_points:,}")
        
        # Agreguj klasy
        class_counts = {}
        for s in all_stats:
            for lbl, cnt in s["labels"].items():
                class_counts[lbl] = class_counts.get(lbl, 0) + cnt
        
        print("\nClass distribution:")
        for lbl in sorted(class_counts.keys()):
            cnt = class_counts[lbl]
            pct = 100 * cnt / total_points if total_points > 0 else 0
            print(f"  Class {lbl:2d}: {cnt:>12,} ({pct:>6.2f}%)")
    
    print("\nDone!")


if __name__ == "__main__":
    main()