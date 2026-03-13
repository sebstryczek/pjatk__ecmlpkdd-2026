# zaha_feature_geometry.py
"""
Obliczanie cech geometrycznych z lokalnego PCA.

Cechy:
- verticality: jak bardzo normalna jest pionowa |nz| ∈ [0, 1]
    - 0 = normalna pozioma (ściana pionowa)
    - 1 = normalna pionowa (teren/dach poziomy)
- planarity: płaskość lokalnego sąsiedztwa (λ2-λ3)/λ1 ∈ [0, 1]
    - wysoka = płaska powierzchnia (ściana, teren)
    - niska = krawędź lub punkt
- linearity: liniowość lokalnego sąsiedztwa (λ1-λ2)/λ1 ∈ [0, 1]
    - wysoka = struktura liniowa (krawędź, rura, gzyms)
    - niska = płaska lub sferyczna
- sphericity: sferyczność λ3/λ1 ∈ [0, 1]
    - wysoka = punkt/sfera
    - niska = płaska lub liniowa
"""
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


def compute_geometric_features(
    coords: np.ndarray,
    normals: np.ndarray = None,
    k_neighbors: int = 30,
    radius: float = 0.3,
    verbose: bool = True
) -> dict:
    """
    Oblicza cechy geometryczne dla każdego punktu.
    
    Args:
        coords: (N, 3) współrzędne
        normals: (N, 3) normalne (opcjonalne - jeśli None, oblicza verticality z PCA)
        k_neighbors: liczba sąsiadów do PCA
        radius: promień wyszukiwania
        verbose: czy wyświetlać postęp
    
    Returns:
        dict z kluczami (wszystkie mają kształt (N, 1) float32):
            - verticality
            - planarity
            - linearity
            - sphericity
    """
    if o3d is None:
        raise RuntimeError("open3d nie jest zainstalowane")
    
    n = coords.shape[0]
    if n == 0:
        return {
            "verticality": np.zeros((0, 1), dtype=np.float32),
            "planarity": np.zeros((0, 1), dtype=np.float32),
            "linearity": np.zeros((0, 1), dtype=np.float32),
            "sphericity": np.zeros((0, 1), dtype=np.float32),
        }
    
    if verbose:
        print(f"[GEOMETRY] Computing features for {n:,} points...", flush=True)
    
    # Buduj KDTree
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64))
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    # Inicjalizuj tablice
    verticality = np.zeros((n, 1), dtype=np.float32)
    planarity = np.zeros((n, 1), dtype=np.float32)
    linearity = np.zeros((n, 1), dtype=np.float32)
    sphericity = np.zeros((n, 1), dtype=np.float32)
    
    # Jeśli mamy normalne - użyj ich do verticality
    if normals is not None:
        if normals.shape[0] != n:
            raise ValueError(f"normals.shape[0]={normals.shape[0]} != coords.shape[0]={n}")
        # Verticality = |nz| (wartość bezwzględna składowej Z normalnej)
        verticality[:, 0] = np.abs(normals[:, 2])
        compute_verticality_from_pca = False
        if verbose:
            print("[GEOMETRY] Using provided normals for verticality.", flush=True)
    else:
        compute_verticality_from_pca = True
        if verbose:
            print("[GEOMETRY] Will compute verticality from local PCA.", flush=True)
    
    if verbose:
        print("[GEOMETRY] Computing local PCA features...", flush=True)
    
    # Progress reporting
    report_interval = max(n // 10, 1000)
    
    for i in range(n):
        if verbose and i > 0 and i % report_interval == 0:
            print(f"[GEOMETRY] Progress: {i:,}/{n:,} ({100*i/n:.1f}%)", flush=True)
        
        # Znajdź sąsiadów (hybrid = radius AND max k)
        [k_found, idx, _] = kdtree.search_hybrid_vector_3d(
            coords[i], radius, k_neighbors
        )
        
        if k_found < 4:
            # Za mało sąsiadów - zostaw domyślne wartości (0)
            continue
        
        neighbors = coords[idx, :]
        
        # === Lokalne PCA ===
        centroid = neighbors.mean(axis=0)
        centered = neighbors - centroid
        
        # Macierz kowariancji (3x3)
        cov = (centered.T @ centered) / k_found
        
        try:
            # Eigenvalue decomposition
            # eigh zwraca eigenvalues w kolejności ROSNĄCEJ
            eigvals, eigvecs = np.linalg.eigh(cov)
            
            # Sortuj MALEJĄCO: λ1 >= λ2 >= λ3
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            
            l1 = eigvals[0]
            l2 = eigvals[1]
            l3 = eigvals[2]
            
            # Zabezpieczenie przed dzieleniem przez 0
            if l1 < 1e-10:
                continue
            
            # === Cechy geometryczne ===
            # Planarity: wysoka gdy λ2 >> λ3 (płaska powierzchnia)
            planarity[i, 0] = (l2 - l3) / l1
            
            # Linearity: wysoka gdy λ1 >> λ2 (struktura liniowa)
            linearity[i, 0] = (l1 - l2) / l1
            
            # Sphericity: wysoka gdy λ1 ≈ λ2 ≈ λ3
            sphericity[i, 0] = l3 / l1
            
            # Verticality z PCA (jeśli nie mamy normalnych)
            if compute_verticality_from_pca:
                # Najmniejszy eigenvector (kolumna 2) = normalna do płaszczyzny
                local_normal = eigvecs[:, 2]
                # Verticality = |nz|
                verticality[i, 0] = np.abs(local_normal[2])
            
        except np.linalg.LinAlgError:
            continue
    
    if verbose:
        print("[GEOMETRY] Done.", flush=True)
        print(f"[GEOMETRY] Statistics:", flush=True)
        print(f"  verticality: min={verticality.min():.3f}, max={verticality.max():.3f}, mean={verticality.mean():.3f}", flush=True)
        print(f"  planarity:   min={planarity.min():.3f}, max={planarity.max():.3f}, mean={planarity.mean():.3f}", flush=True)
        print(f"  linearity:   min={linearity.min():.3f}, max={linearity.max():.3f}, mean={linearity.mean():.3f}", flush=True)
        print(f"  sphericity:  min={sphericity.min():.3f}, max={sphericity.max():.3f}, mean={sphericity.mean():.3f}", flush=True)
    
    return {
        "verticality": verticality,
        "planarity": planarity,
        "linearity": linearity,
        "sphericity": sphericity,
    }