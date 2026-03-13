import numpy as np
from .transform import TRANSFORMS

@TRANSFORMS.register_module()
class ClassAwareCropKNN:
    """
    Class-aware crop bez preprocessingu:
    - wybiera centrum cropa z target klas (np. Door/Blinds) z prawdopodobieństwem p_target
    - wycina point_max najbliższych punktów (kNN po dystansie)
    """

    def __init__(
        self,
        point_max: int,
        target_classes,
        p_target: float = 0.7,
        min_target_points: int = 1,
    ):
        self.point_max = int(point_max)
        self.target_classes = np.array(list(target_classes), dtype=np.int64)
        self.p_target = float(p_target)
        self.min_target_points = int(min_target_points)

        assert self.point_max > 0
        assert 0.0 <= self.p_target <= 1.0

    def __call__(self, data_dict):
        if "coord" not in data_dict:
            return data_dict

        coord = data_dict["coord"]
        n = coord.shape[0]
        if n <= self.point_max:
            return data_dict

        seg = data_dict.get("segment", None)

        # --- wybór centrum ---
        use_target = False
        if seg is not None:
            seg = seg.reshape(-1)
            mask = np.isin(seg, self.target_classes)
            target_idx = np.nonzero(mask)[0]
            if target_idx.size >= self.min_target_points and (np.random.rand() < self.p_target):
                center_idx = np.random.choice(target_idx)
                use_target = True
            else:
                center_idx = np.random.randint(0, n)
        else:
            center_idx = np.random.randint(0, n)

        center = coord[center_idx]

        # --- kNN crop ---
        # dystans kwadratowy (bez sqrt)
        d2 = np.sum((coord - center) ** 2, axis=1)

        # szybki wybór point_max najmniejszych
        idx = np.argpartition(d2, self.point_max - 1)[: self.point_max]

        # (opcjonalnie) możesz przetasować, żeby nie było zawsze „od najbliższych”
        np.random.shuffle(idx)

        # --- przytnij wszystkie tablice o długości N ---
        for k, v in list(data_dict.items()):
            if isinstance(v, np.ndarray) and v.shape[0] == n:
                data_dict[k] = v[idx]

        # debug/analiza (opcjonalnie)
        data_dict["__class_aware_used_target__"] = np.array([1 if use_target else 0], dtype=np.int32)

        return data_dict


@TRANSFORMS.register_module()
class BoundaryMaskFromGrid:
    """
    Wyznacza boundary points na bazie grid_coord.
    Punkt jest boundary, jeśli w sąsiedztwie istnieje punkt o innej klasie.
    
    Args:
        ignore_index (int): Indeks ignorowany w segmentacji.
        connectivity (int): 6 (faces) lub 26 (faces + edges + corners).
    """

    def __init__(self, ignore_index=-1, connectivity=6):
        assert connectivity in (6, 26), "connectivity must be 6 or 26"
        self.ignore_index = int(ignore_index)
        self.connectivity = int(connectivity)

        if self.connectivity == 6:
            # 6-neighborhood: tylko ściany
            self.offsets = np.array([
                [ 1, 0, 0], [-1, 0, 0],
                [ 0, 1, 0], [ 0,-1, 0],
                [ 0, 0, 1], [ 0, 0,-1],
            ], dtype=np.int32)
        else:
            # 26-neighborhood: ściany + krawędzie + rogi
            offs = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == dy == dz == 0:
                            continue
                        offs.append([dx, dy, dz])
            self.offsets = np.array(offs, dtype=np.int32)

    def __call__(self, data_dict):
        # Sprawdź czy mamy potrzebne dane
        if "grid_coord" not in data_dict or "segment" not in data_dict:
            data_dict["boundary"] = np.zeros(1, dtype=np.int32)
            return data_dict

        gc = data_dict["grid_coord"].astype(np.int32)
        seg = data_dict["segment"].reshape(-1).astype(np.int32)
        n = seg.shape[0]

        # Inicjalizuj boundary mask
        boundary = np.zeros(n, dtype=np.int32)
        
        # Valid points (nie ignore_index)
        valid = seg != self.ignore_index
        if not np.any(valid):
            data_dict["boundary"] = boundary
            return data_dict

        # Hashmap: voxel coordinate -> label
        voxel2label = {}
        for idx in np.where(valid)[0]:
            key = (int(gc[idx, 0]), int(gc[idx, 1]), int(gc[idx, 2]))
            voxel2label[key] = int(seg[idx])

        # Sprawdź każdy valid punkt
        for i in np.where(valid)[0]:
            vx = (int(gc[i, 0]), int(gc[i, 1]), int(gc[i, 2]))
            current_label = int(seg[i])
            
            # Sprawdź sąsiadów
            for off in self.offsets:
                neighbor = (vx[0] + int(off[0]), vx[1] + int(off[1]), vx[2] + int(off[2]))
                neighbor_label = voxel2label.get(neighbor, None)
                
                # Jeśli sąsiad ma inną klasę - punkt jest graniczny
                if neighbor_label is not None and neighbor_label != current_label:
                    boundary[i] = 1
                    break

        data_dict["boundary"] = boundary
        return data_dict

import numpy as np
from pointcept.datasets.transform import TRANSFORMS, index_operator


import numpy as np
from .transform import TRANSFORMS  # registry

@TRANSFORMS.register_module()
class GridCrop:
    """
    Grid-based crop z równomiernym pokryciem XY + opcjonalny edge-bias.
    Wybiera losową *zajętą* komórkę siatki i zwraca crop o stałej liczbie punktów.

    Ulepszenia vs prosta wersja:
    - retry-loop (max_tries) żeby unikać ubogich komórek
    - "cell" mode: zawsze zachowuje punkty z komórki + dopełnia kNN bez duplikatów
    - poprawiona logika brzegów: sel_ix >= nx-edge_width (i analogicznie dla Y)
    - opcjonalne preferowanie target klas (domyślnie wyłączone)

    Args:
        cell_size (float): rozmiar komórki siatki w metrach (XY)
        point_max (int): liczba punktów po cropie
        min_points_in_cell (int): minimalna liczba punktów, by komórka była kandydatem
        p_edge (float): prawdopodobieństwo wyboru komórki brzegowej
        edge_width (int): ile komórek od krawędzi uznajemy za brzeg
        sample_mode (str): "knn" lub "cell"
        shuffle (bool): przetasuj indeksy
        max_tries (int): ile razy próbować znaleźć sensowną komórkę (fallback na losowy kNN)
        prefer_target_classes (tuple/list/int): klasy, które (opcjonalnie) preferujemy w wyborze
        p_target_cell (float): P(wybierz komórkę zawierającą target klasę), jeśli dostępna
        p_target_center (float): P(wybierz centrum z target punktu w wybranej komórce)
        debug (bool): dopisz do data_dict metadane diagnostyczne
    """

    def __init__(
        self,
        cell_size=10.0,
        point_max=40960,
        min_points_in_cell=64,
        p_edge=0.3,
        edge_width=1,
        sample_mode="knn",
        shuffle=True,
        max_tries=10,
        prefer_target_classes=None,
        p_target_cell=0.0,
        p_target_center=0.0,
        debug=False,
    ):
        self.cell_size = float(cell_size)
        self.point_max = int(point_max)
        self.min_points_in_cell = int(min_points_in_cell)
        self.p_edge = float(p_edge)
        self.edge_width = int(edge_width)
        assert sample_mode in ("knn", "cell")
        self.sample_mode = sample_mode
        self.shuffle = bool(shuffle)
        self.max_tries = int(max_tries)
        self.debug = bool(debug)

        if prefer_target_classes is None:
            self.prefer_target_classes = None
        else:
            if isinstance(prefer_target_classes, (int, np.integer)):
                prefer_target_classes = [int(prefer_target_classes)]
            self.prefer_target_classes = np.array(list(prefer_target_classes), dtype=np.int32)

        self.p_target_cell = float(p_target_cell)
        self.p_target_center = float(p_target_center)

        assert self.point_max > 0
        assert self.cell_size > 0
        assert 0.0 <= self.p_edge <= 1.0
        assert 0.0 <= self.p_target_cell <= 1.0
        assert 0.0 <= self.p_target_center <= 1.0

    @staticmethod
    def _knn_indices(coord, center, k):
        """Zwraca k indeksów najbliższych (argpartition, bez sortowania)."""
        d2 = np.sum((coord - center) ** 2, axis=1)
        k = int(min(k, coord.shape[0]))
        idx = np.argpartition(d2, k - 1)[:k]
        return idx

    def __call__(self, data_dict):
        if "coord" not in data_dict:
            return data_dict

        coord = data_dict["coord"]
        n = coord.shape[0]
        if n <= self.point_max:
            return data_dict

        # segment (opcjonalnie) do target preferencji
        seg = data_dict.get("segment", None)
        if seg is not None:
            seg = np.asarray(seg).reshape(-1)

        # --- komórki XY ---
        xy = coord[:, :2]
        xy_min = xy.min(axis=0)
        cell_xy = np.floor((xy - xy_min) / self.cell_size).astype(np.int32)
        ix = cell_xy[:, 0]
        iy = cell_xy[:, 1]

        # ravel-hash bez kolizji (min jest 0, bo odejmujemy xy_min)
        iy_max = int(iy.max())
        key = ix.astype(np.int64) * (iy_max + 1) + iy.astype(np.int64)

        # grupowanie punktów po komórkach
        order = np.argsort(key)
        key_sorted = key[order]
        unique_keys, start_idx, counts = np.unique(
            key_sorted, return_index=True, return_counts=True
        )

        # filtr: komórki z min_points_in_cell (fallback na >=1)
        valid = counts >= self.min_points_in_cell
        if not np.any(valid):
            valid = counts >= 1

        unique_keys = unique_keys[valid]
        start_idx = start_idx[valid]
        counts = counts[valid]

        if unique_keys.size == 0:
            # skrajny fallback
            idx_keep = np.random.choice(n, self.point_max, replace=False)
            from pointcept.datasets.transform import index_operator
            return index_operator(data_dict, idx_keep)

        # (ix,iy) dla wybranych komórek
        sel_ix = (unique_keys // (iy_max + 1)).astype(np.int32)
        sel_iy = (unique_keys % (iy_max + 1)).astype(np.int32)

        # rozmiar siatki (liczba komórek w X i Y)
        nx = int(ix.max()) + 1
        ny = int(iy.max()) + 1

        # --- edge mask ---
        if self.edge_width > 0:
            ew = self.edge_width
            edge_mask = (
                (sel_ix < ew) | (sel_ix >= nx - ew) |
                (sel_iy < ew) | (sel_iy >= ny - ew)
            )
        else:
            edge_mask = np.zeros_like(sel_ix, dtype=bool)

        # opcjonalna preferencja komórek zawierających target klasy
        target_cell_mask = None
        if self.prefer_target_classes is not None and seg is not None and unique_keys.size > 0:
            target_cell_mask = np.zeros(unique_keys.shape[0], dtype=bool)
            for ci in range(unique_keys.shape[0]):
                s = start_idx[ci]
                c = counts[ci]
                cell_pts = order[s:s+c]
                if np.any(np.isin(seg[cell_pts], self.prefer_target_classes)):
                    target_cell_mask[ci] = True

        used_edge = False
        used_target_cell = False
        attempts = 0

        chosen_cell_idx = None
        for attempts in range(1, self.max_tries + 1):
            # 1) wybór: edge vs normal
            if np.any(edge_mask) and (np.random.rand() < self.p_edge):
                cand = np.where(edge_mask)[0]
                cell_idx = int(np.random.choice(cand))
                used_edge = True
            else:
                cell_idx = int(np.random.randint(0, unique_keys.shape[0]))

            # 2) (opcjonalnie) preferuj target-cell
            if target_cell_mask is not None and np.any(target_cell_mask) and (np.random.rand() < self.p_target_cell):
                cand_t = np.where(target_cell_mask)[0]
                cell_idx = int(np.random.choice(cand_t))
                used_target_cell = True

            # sanity: cell ma przynajmniej 1 punkt
            if counts[cell_idx] > 0:
                chosen_cell_idx = cell_idx
                break

        if chosen_cell_idx is None:
            chosen_cell_idx = int(np.random.randint(0, unique_keys.shape[0]))

        # indeksy punktów w wybranej komórce
        s = start_idx[chosen_cell_idx]
        c = counts[chosen_cell_idx]
        cell_point_indices = order[s:s+c]

        # wybór centrum (opcjonalnie preferuj target punkt)
        if self.prefer_target_classes is not None and seg is not None and (np.random.rand() < self.p_target_center):
            mask_t = np.isin(seg[cell_point_indices], self.prefer_target_classes)
            t_idx = cell_point_indices[mask_t]
            if t_idx.size > 0:
                center_idx = int(np.random.choice(t_idx))
            else:
                center_idx = int(np.random.choice(cell_point_indices))
        else:
            center_idx = int(np.random.choice(cell_point_indices))

        center = coord[center_idx]

        # --- finalne idx ---
        if self.sample_mode == "knn":
            idx_keep = self._knn_indices(coord, center, self.point_max)

        else:  # "cell"
            if cell_point_indices.size >= self.point_max:
                idx_keep = np.random.choice(cell_point_indices, self.point_max, replace=False)
            else:
                # 1) zachowaj WSZYSTKO z komórki
                base = cell_point_indices

                # 2) dopełnij kNN wokół centrum (weź nadmiar, usuń duplikaty)
                need = self.point_max - base.size

                # bierzemy trochę więcej, żeby po usunięciu duplikatów nadal starczyło
                k_try = min(n, max(self.point_max * 4, base.size + need))
                knn = self._knn_indices(coord, center, k_try)

                # usuń duplikaty z base
                base_set = np.zeros(n, dtype=bool)
                base_set[base] = True
                knn = knn[~base_set[knn]]

                if knn.size >= need:
                    fill = knn[:need]
                else:
                    # jeśli wciąż brakuje, dopełnij losowo z reszty (bez powtórek)
                    remaining = np.where(~base_set)[0]
                    # usuń już użyte w knn
                    tmp = np.zeros(n, dtype=bool)
                    tmp[base] = True
                    tmp[knn] = True
                    remaining = np.where(~tmp)[0]
                    if remaining.size > 0:
                        extra_need = need - knn.size
                        extra_need = min(extra_need, remaining.size)
                        extra = np.random.choice(remaining, extra_need, replace=False)
                        fill = np.concatenate([knn, extra], axis=0)
                    else:
                        fill = knn

                idx_keep = np.concatenate([base, fill], axis=0)

                # jeśli przez jakieś skrajności nadal nie mamy point_max, fallback
                if idx_keep.size < self.point_max:
                    missing = self.point_max - idx_keep.size
                    pool = np.setdiff1d(np.arange(n), idx_keep, assume_unique=False)
                    if pool.size >= missing:
                        extra = np.random.choice(pool, missing, replace=False)
                        idx_keep = np.concatenate([idx_keep, extra], axis=0)

                # ostateczna korekta rozmiaru
                if idx_keep.size > self.point_max:
                    idx_keep = idx_keep[:self.point_max]

        if self.shuffle:
            np.random.shuffle(idx_keep)

        # przytnij wszystkie pola zgodnie z Pointcept
        from pointcept.datasets.transform import index_operator
        data_dict = index_operator(data_dict, idx_keep)

        if self.debug:
            data_dict["__gridcrop_used_edge__"] = np.array([1 if used_edge else 0], dtype=np.int32)
            data_dict["__gridcrop_used_target_cell__"] = np.array([1 if used_target_cell else 0], dtype=np.int32)
            data_dict["__gridcrop_attempts__"] = np.array([attempts], dtype=np.int32)
            data_dict["__gridcrop_cell__"] = np.array(
                [int(sel_ix[chosen_cell_idx]), int(sel_iy[chosen_cell_idx])], dtype=np.int32
            )
            data_dict["__gridcrop_cell_count__"] = np.array([int(c)], dtype=np.int32)

        return data_dict
