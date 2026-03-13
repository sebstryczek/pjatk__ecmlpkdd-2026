_base_ = ["../_base_/default_runtime.py"]
seed = 42
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    # dict(type="PreciseEvaluator", test_last=False),
]

CLASSES = [
    "void",           # 0
    "wall",           # 1
    "window",         # 2
    "door",           # 3
    "balcony",        # 4
    "molding",        # 5
    "deco",           # 6
    "column",         # 7
    "arch",           # 8
    "stairs",         # 9
    "ground surface", # 10
    "terrain",        # 11
    "roof",           # 12
    "blinds",         # 13
    "interior",       # 14
    "other",          # 15
]
NUMBER_OF_CLASSES = len(CLASSES)
IGNORE_INDEX = 0

# Adding more features reduce results
FEATURE_KEYS = (
    "coord",        # (N, 3) - zawsze
    # "normal",       # (N, 3)
    # "height",       # (N, 1)
    # "verticality",  # (N, 1)
    # "planarity",    # (N, 1)
    # "linearity",    # (N, 1)
    # "sphericity", # opcjonalnie
    # "displacement", # (N, 3)
)

FEATURE_CHANNELS = sum([
    3,  # coord
    # 3,  # normal
    # 1,  # height
    # 1,  # verticality
    # 1,  # planarity
    # 1,  # linearity
    # 1,  # sphericity
    # 3,  # displacement
])

"""
ZAHA LoFG3: Point Transformer v1 — reproduction of Wysocki et al. (WACV 2025)
===============================================================================

Epoch semantics:
  ZAHADataset.__len__ = total_training_points / 1024 ≈ 390,000
  epoch=100, eval_epoch=100 → loop=1
  One eval_epoch = one pass through ~390k samples = one "ZAHA epoch"
  100 eval_epochs = 100 ZAHA epochs

  Total: ~390k * 100 = ~39M samples, batch_size=32 → ~1.2M iterations
  (matching yanx27-style training used in the ZAHA paper)

Sampling:
  ZAHA replaced PT's voxelization with "index segmentation" — the same
  point-proportional sampling as yanx27's PointNet/PointNet++ implementation.
  ZAHADataset implements this: each facade generates samples proportional
  to its point count. SphereCrop(point_max=1024) selects spatially coherent
  patches, which is a reasonable approximation.

Features:
  Only XYZ coordinates (in_channels=3), matching ZAHA paper.
"""


# =============================================================================
# Training schedule
# =============================================================================
# epoch / eval_epoch = loop (auto-set by framework)
# 100 / 100 = 1 → __len__ = num_samples * 1
# ZAHADataset.__len__ ≈ 390,000 (from total_points / 1024)
# 100 eval_epochs × 390,000 samples = 39M samples total
epoch = 100
eval_epoch = 100

# =============================================================================
# Misc
# =============================================================================
batch_size = 32          # ZAHA paper
mix_prob = 0.0
empty_cache = False
enable_amp = True

# =============================================================================
# Model — Point Transformer v1
# =============================================================================
model = dict(
    type="DefaultSegmentor",
    backbone=dict(
        type="PointTransformer-Seg26",
        in_channels=FEATURE_CHANNELS,       # only XYZ
        num_classes=NUMBER_OF_CLASSES,       # LoFG3
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=IGNORE_INDEX),
    ],
)

# =============================================================================
# Optimizer & Scheduler — ZAHA supplement A.2
# =============================================================================
optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001)
scheduler = dict(
    type="MultiStepLR",
    milestones=[0.6, 0.8],   # lr: 0.1 → 0.01 @epoch60, → 0.001 @epoch80
    gamma=0.1,
)

# =============================================================================
# Dataset
# =============================================================================
dataset_type = "ZAHADataset"
data_root = "data/zaha"

data = dict(
    num_classes=NUMBER_OF_CLASSES,
    ignore_index=IGNORE_INDEX,
    names=CLASSES,

    # -----------------------------------------------------------------
    # TRAIN
    # -----------------------------------------------------------------
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        num_point=1024,        # ZAHA paper: 1024 points per sample
        sample_rate=1.0,       # full coverage per epoch
        pre_crop_size=1024 * 16,    # cheap spatial crop before transforms (memory optimization)
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=1024, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
    ),

    # -----------------------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------------------
    val=dict(
        type=dataset_type,
        split="validation",
        data_root=data_root,
        pre_crop_size=1024 * 16,    # cheap spatial crop before transforms (memory optimization)
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.001,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "origin_segment",
                    "inverse",
                ),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
    ),

    # -----------------------------------------------------------------
    # TEST
    # -----------------------------------------------------------------
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord",),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1, 1]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
    ),
)