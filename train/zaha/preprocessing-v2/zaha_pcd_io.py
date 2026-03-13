# zaha_pcd_io.py
"""
Moduł do wczytywania plików PCD z datasetu ZAHA.
"""
import numpy as np


def _print_level_2(msg: str) -> None:
    print(f"  [IO] {msg}", flush=True)


def load_pcd_with_labels(pcd_path: str):
    """
    Wczytuje plik .pcd, zwraca:
      - coords: (N, 3) float32
      - colors: (N, 3) uint8 lub None (jeśli r/g/b brak)
      - labels: (N,) int32
    """
    _print_level_2(f"Parsing PCD header: {pcd_path}")

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
        raise RuntimeError(f"Nieprawidłowy nagłówek PCD w pliku: {pcd_path}")

    parts = fields_line.split()
    fields = [f.lower() for f in parts[1:]]
    _print_level_2(f"Fields: {fields}")

    _print_level_2("Loading point data...")
    data = np.loadtxt(pcd_path, skiprows=data_start_idx)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != len(fields):
        raise RuntimeError(
            f"Plik {pcd_path}: liczba kolumn ({data.shape[1]}) "
            f"nie zgadza się z liczbą pól FIELDS ({len(fields)})."
        )

    field_to_col = {name: idx for idx, name in enumerate(fields)}
    _print_level_2(f"Loaded data shape: {data.shape}")

    # Współrzędne (wymagane)
    try:
        x_idx = field_to_col["x"]
        y_idx = field_to_col["y"]
        z_idx = field_to_col["z"]
    except KeyError as e:
        raise RuntimeError(f"Brak pola {e} w FIELDS dla pliku: {pcd_path}")

    coords = data[:, [x_idx, y_idx, z_idx]].astype(np.float32)

    # Kolory (opcjonalne)
    if all(k in field_to_col for k in ("r", "g", "b")):
        r_idx = field_to_col["r"]
        g_idx = field_to_col["g"]
        b_idx = field_to_col["b"]
        colors = data[:, [r_idx, g_idx, b_idx]].astype(np.float32)
        colors = np.clip(colors, 0, 255).astype(np.uint8)
    else:
        colors = None

    # Etykiety
    label_field_candidates = ["label", "class", "classification"]
    label_idx = None
    for name in label_field_candidates:
        if name in field_to_col:
            label_idx = field_to_col[name]
            break

    if label_idx is not None:
        labels = data[:, label_idx].astype(np.int32)
    else:
        _print_level_2("WARNING: brak pola label - ustawiam labels=0")
        labels = np.zeros((coords.shape[0],), dtype=np.int32)

    _print_level_2("Finished loading PCD.")
    return coords, colors, labels