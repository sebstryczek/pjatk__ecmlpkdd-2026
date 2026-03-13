# zaha_feature_height.py
"""
Obliczanie cechy wysokości dla punktów.
"""
import numpy as np


def compute_height_feature(coords: np.ndarray, mode: str = "z_norm01") -> np.ndarray:
    """
    Oblicza cechę wysokości dla każdego punktu.
    
    Args:
        coords: (N, 3) - współrzędne (oś Z = wysokość)
        mode: 
            - "z_raw": surowa wartość Z
            - "z_min": h = z - min(z), wartości >= 0
            - "z_norm01": h = (z - min(z)) / (max(z) - min(z)), wartości w [0, 1]
    
    Returns:
        height: (N, 1) float32
    """
    if coords.shape[0] == 0:
        return np.zeros((0, 1), dtype=np.float32)
    
    z = coords[:, 2].astype(np.float64)  # float64 dla precyzji
    
    if mode == "z_raw":
        height = z
    elif mode == "z_min":
        z_min = z.min()
        height = z - z_min
    elif mode == "z_norm01":
        z_min = z.min()
        z_max = z.max()
        z_range = z_max - z_min
        
        if z_range < 1e-6:
            # Wszystkie punkty na tej samej wysokości
            print("[HEIGHT] WARNING: z_range < 1e-6, setting height to 0.5", flush=True)
            height = np.full_like(z, 0.5)
        else:
            height = (z - z_min) / z_range
    else:
        raise ValueError(f"Nieznany mode='{mode}'. Dostępne: z_raw, z_min, z_norm01")

    return height.astype(np.float32).reshape(-1, 1)