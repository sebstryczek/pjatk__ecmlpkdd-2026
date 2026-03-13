

# ----- v3
import os
import numpy as np
from .defaults import DefaultDataset
from .builder import DATASETS


@DATASETS.register_module()
class ZAHADataset(DefaultDataset):
    # rozszerzamy listę dozwolonych assetów, żeby DefaultDataset je wczytał
    VALID_ASSETS = DefaultDataset.VALID_ASSETS + [
        "height",
        "verticality",
        "planarity",
        "linearity",
        "sphericity",
    ]


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_data_name(self, idx):
        scene_dir = self.data_list[idx % len(self.data_list)]
        scene_name = os.path.basename(scene_dir)
        split_name = os.path.basename(os.path.dirname(scene_dir))
        return f"{split_name}-{scene_name}"

    def get_data(self, idx):
        # wczytanie zgodne z DefaultDataset (bez mapowania segmentów)
        data_dict = super().get_data(idx)

        # opcjonalnie: jak wcześniej, trzymaj cechy geometryczne jako float32
        for k in ["height", "verticality", "planarity", "linearity", "sphericity"]:
            if k in data_dict:
                data_dict[k] = data_dict[k].astype(np.float32)

        return data_dict
        
    def prepare_test_data(self, idx):
        from copy import deepcopy

        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)

        # --- OVERRIDE: zachowaj origin_segment przed voxelizacją ---
        name = data_dict.pop("name")
        origin_segment = data_dict["segment"].copy()
        # --- END OVERRIDE ---

        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))
        del data_dict

        fragment_list = []
        inverse = None  # OVERRIDE: zbieramy inverse z voxelizacji
        for data in data_dict_list:
            if self.test_voxelize is not None:
                result = self.test_voxelize(data)
                # OVERRIDE: mode="train" zwraca dict, nie listę
                data_part_list = result if isinstance(result, list) else [result]
            else:
                data["index"] = np.arange(data["coord"].shape[0])
                data_part_list = [data]

            for dp in data_part_list:
                # --- OVERRIDE: wyciągnij inverse, usuń segment, ustaw index ---
                if "inverse" in dp and inverse is None:
                    inverse = dp.pop("inverse")
                elif "inverse" in dp:
                    dp.pop("inverse")
                if "segment" in dp:
                    dp.pop("segment")
                dp["index"] = np.arange(dp["coord"].shape[0])
                # --- END OVERRIDE ---

                if self.test_crop is not None:
                    crops = self.test_crop(dp)
                    fragment_list += crops if isinstance(crops, list) else [crops]
                else:
                    fragment_list.append(dp)

        for i in range(len(fragment_list)):
            fragment_list[i] = self.post_transform(fragment_list[i])

        # --- OVERRIDE: result_dict z origin_segment + inverse do mapowania na oryginalne punkty ---
        vox_size = fragment_list[0]["coord"].shape[0] if fragment_list else 0
        return dict(
            segment=np.zeros(vox_size, dtype=np.int32),  # placeholder, pred alokuje się na tym rozmiarze
            name=name,
            origin_segment=origin_segment,
            inverse=inverse,
            fragment_list=fragment_list,
        )
        # --- END OVERRIDE ---
# ----



# 
# 
# 
# 





"""
V1
ZAHA Dataset for Pointcept

Implements yanx27-style sampling to reproduce ZAHA paper experiments.
Each facade generates multiple samples proportional to its point count.

Memory/speed optimization:
  - Only coord and segment are preloaded
  - KDTree built once per facade at init for O(log N) spatial queries
  - __getitem__ does fast KDTree query instead of brute-force distances

Reference: yanx27/Pointnet_Pointnet2_pytorch/data_utils/S3DISDataLoader.py
"""

# import os
# import numpy as np
# from scipy.spatial import cKDTree

# from .defaults import DefaultDataset
# from .builder import DATASETS
# from pointcept.utils.logger import get_root_logger


# @DATASETS.register_module()
# class ZAHADataset(DefaultDataset):
#     VALID_ASSETS = DefaultDataset.VALID_ASSETS + [
#         "height",
#         "verticality",
#         "planarity",
#         "linearity",
#         "sphericity",
#     ]

#     def __init__(
#         self,
#         num_point=1024,
#         sample_rate=1.0,
#         pre_crop_size=4096,
#         *args,
#         **kwargs,
#     ):
#         self.num_point = num_point
#         self.sample_rate = sample_rate
#         self.pre_crop_size = pre_crop_size
#         self.facade_coords = []
#         self.facade_segments = []
#         self.facade_trees = []
#         self.facade_idxs = np.array([])

#         super().__init__(*args, **kwargs)

#         self._preload_facades()
#         if not self.test_mode:
#             self._build_sample_index()

#     def _preload_facades(self):
#         """Load coord+segment into memory and build KDTree per facade."""
#         logger = get_root_logger()
#         total_points = 0

#         for i, data_path in enumerate(self.data_list):
#             coord = np.load(
#                 os.path.join(data_path, "coord.npy")
#             ).astype(np.float32)
#             seg_path = os.path.join(data_path, "segment.npy")
#             if os.path.exists(seg_path):
#                 segment = np.load(seg_path).reshape([-1]).astype(np.int32)
#             else:
#                 segment = np.ones(coord.shape[0], dtype=np.int32) * -1

#             n_points = coord.shape[0]
#             total_points += n_points
#             self.facade_coords.append(coord)
#             self.facade_segments.append(segment)

#             if not self.test_mode:
#                 logger.info(
#                     f"  Building KDTree for facade {i+1}/{len(self.data_list)} "
#                     f"({os.path.basename(data_path)}, {n_points:,} pts)..."
#                 )
#                 tree = cKDTree(coord)
#                 self.facade_trees.append(tree)

#         logger.info(
#             f"ZAHADataset: preloaded {len(self.facade_coords)} facades, "
#             f"{total_points:,} total points (coord+segment only, "
#             f"~{total_points * 16 / 1e9:.1f} GB)"
#         )

#     def _build_sample_index(self):
#         """Build yanx27-style proportional sampling index."""
#         logger = get_root_logger()

#         num_point_all = np.array(
#             [c.shape[0] for c in self.facade_coords], dtype=np.float64
#         )
#         total_points = np.sum(num_point_all)
#         num_iter = int(total_points * self.sample_rate / self.num_point)

#         sample_prob = num_point_all / total_points
#         facade_idxs = []
#         for i in range(len(self.facade_coords)):
#             count = int(round(sample_prob[i] * num_iter))
#             facade_idxs.extend([i] * count)
#         self.facade_idxs = np.array(facade_idxs)

#         logger.info(
#             f"ZAHADataset: {len(self.facade_coords)} facades, "
#             f"{int(total_points):,} total points, "
#             f"{len(self.facade_idxs):,} samples/epoch "
#             f"(~{self.num_point} pts/sample, sample_rate={self.sample_rate})"
#         )

#     def _spatial_crop(self, facade_idx, n_out, min_extent=0.5, max_retries=10):
#         coord = self.facade_coords[facade_idx]
#         segment = self.facade_segments[facade_idx]
#         tree = self.facade_trees[facade_idx]
#         N = coord.shape[0]

#         if N <= n_out:
#             return coord.copy(), segment.copy()

#         for _ in range(max_retries):
#             center = coord[np.random.randint(N)]
#             _, crop_idx = tree.query(center, k=n_out)
#             cropped = coord[crop_idx]
#             extent = cropped.max(0) - cropped.min(0)
#             if extent.min() > min_extent:
#                 return cropped.copy(), segment[crop_idx].copy()

#         # Fallback: return last attempt anyway
#         return cropped.copy(), segment[crop_idx].copy()

#     def get_data_name(self, idx):
#         if self.test_mode:
#             data_path = self.data_list[idx % len(self.data_list)]
#         else:
#             facade_idx = self.facade_idxs[idx % len(self.facade_idxs)]
#             data_path = self.data_list[facade_idx]
#         return os.path.basename(data_path)

#     def get_data(self, idx):
#         if self.test_mode:
#             return self._load_full_facade(idx % len(self.data_list))
#         else:
#             facade_idx = self.facade_idxs[idx % len(self.facade_idxs)]
#             coord, segment = self._spatial_crop(facade_idx, self.pre_crop_size)
#             data_path = self.data_list[facade_idx]
#             return {
#                 "coord": coord,
#                 "segment": segment,
#                 "name": os.path.basename(data_path),
#                 "split": os.path.basename(os.path.dirname(data_path)),
#                 "instance": np.ones(coord.shape[0], dtype=np.int32) * -1,
#             }

#     def _load_full_facade(self, facade_idx):
#         """Load full facade from disk for test/val mode."""
#         data_path = self.data_list[facade_idx]
#         data_dict = {}
#         assets = os.listdir(data_path)
#         for asset in assets:
#             if not asset.endswith(".npy"):
#                 continue
#             if asset[:-4] not in self.VALID_ASSETS:
#                 continue
#             data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))

#         if "coord" in data_dict:
#             data_dict["coord"] = data_dict["coord"].astype(np.float32)
#         if "color" in data_dict:
#             data_dict["color"] = data_dict["color"].astype(np.float32)
#         if "normal" in data_dict:
#             data_dict["normal"] = data_dict["normal"].astype(np.float32)
#         for k in ["height", "verticality", "planarity", "linearity", "sphericity"]:
#             if k in data_dict:
#                 data_dict[k] = data_dict[k].astype(np.float32)
#         if "segment" in data_dict:
#             data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
#         else:
#             data_dict["segment"] = (
#                 np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
#             )
#         if "instance" not in data_dict:
#             data_dict["instance"] = (
#                 np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
#             )

#         data_dict["name"] = os.path.basename(data_path)
#         data_dict["split"] = os.path.basename(os.path.dirname(data_path))
#         return data_dict

#     def __len__(self):
#         if self.test_mode:
#             return len(self.data_list)
#         else:
#             return len(self.facade_idxs) * self.loop

            