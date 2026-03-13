# zaha_feature_normals.py
"""
Obliczanie normalnych powierzchni dla chmury punktów.
Normalne są orientowane NA ZEWNĄTRZ od centroidu (dla skanów wokół budynku).
"""
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


# Parametry domyślne
NORMAL_RADIUS = 0.3
NORMAL_MAX_NN = 30


def orient_normals_outward(coords: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    Orientuje normalne na zewnątrz od centroidu chmury punktów.
    
    Dla budynków skanowanych wokół:
    - Centroid jest "wewnątrz" budynku
    - Normalne powinny wskazywać "na zewnątrz" (w kierunku skanera)
    
    Args:
        coords: (N, 3) współrzędne punktów
        normals: (N, 3) normalne (mogą być w dowolnym kierunku)
    
    Returns:
        normals: (N, 3) normalne zorientowane na zewnątrz
    """
    # Centroid chmury punktów (przybliżony "środek" budynku)
    centroid = coords.mean(axis=0)
    
    # Wektor od centroidu do każdego punktu (kierunek "na zewnątrz")
    outward = coords - centroid
    
    # Normalizuj wektory outward
    outward_norm = np.linalg.norm(outward, axis=1, keepdims=True)
    outward_norm = np.maximum(outward_norm, 1e-8)
    outward = outward / outward_norm
    
    # Iloczyn skalarny: normalna · outward
    # Jeśli dodatni: normalna wskazuje na zewnątrz (OK)
    # Jeśli ujemny: normalna wskazuje do wewnątrz (trzeba odwrócić)
    dot = np.sum(normals * outward, axis=1)
    
    # Odwróć normalne które wskazują do wewnątrz
    flip_mask = dot < 0
    normals[flip_mask] *= -1
    
    n_flipped = flip_mask.sum()
    pct_flipped = 100 * n_flipped / len(normals)
    print(f"[NORMALS] Flipped {n_flipped:,} / {len(normals):,} normals ({pct_flipped:.1f}%)", flush=True)
    
    return normals


def compute_normals(
    coords: np.ndarray,
    radius: float = NORMAL_RADIUS,
    max_nn: int = NORMAL_MAX_NN,
) -> np.ndarray:
    """
    Oblicza normalne przy użyciu Open3D i orientuje je na zewnątrz.
    
    Args:
        coords: (N, 3) float32 - współrzędne punktów
        radius: promień wyszukiwania sąsiadów
        max_nn: maksymalna liczba sąsiadów
    
    Returns:
        normals: (N, 3) float32 - wektory normalne (znormalizowane, na zewnątrz)
    """
    if o3d is None:
        raise RuntimeError("open3d nie jest zainstalowane. Zainstaluj: pip install open3d")
    
    n_points = coords.shape[0]
    print(f"[NORMALS] Computing normals for {n_points:,} points...", flush=True)

    # Utwórz point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64))

    # Estymacja normalnych
    print(f"[NORMALS] Estimating (radius={radius}, max_nn={max_nn})...", flush=True)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=max_nn,
        )
    )

    # Normalizacja
    pcd.normalize_normals()
    
    # Pobierz normalne
    normals = np.asarray(pcd.normals).astype(np.float32)
    
    # === ORIENTACJA NA ZEWNĄTRZ ===
    print("[NORMALS] Orienting outward from centroid...", flush=True)
    normals = orient_normals_outward(coords, normals)
    
    # Finalna normalizacja
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normals = (normals / norms).astype(np.float32)
    
    # Walidacja
    final_norms = np.linalg.norm(normals, axis=1)
    invalid_count = np.sum(np.abs(final_norms - 1.0) > 0.01)
    if invalid_count > 0:
        print(f"[NORMALS] WARNING: {invalid_count} normals have invalid length!", flush=True)
    
    print("[NORMALS] Done.", flush=True)
    
    return normals