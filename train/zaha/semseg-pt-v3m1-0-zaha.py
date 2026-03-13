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

resume = True
weight = "exp/zaha/semseg-pt-v3m1-0-zaha-exp/model/model_last.pth"

batch_size = 4
batch_size_test = 1
num_worker = 2

mix_prob = 0.8
empty_cache = True
enable_amp = True  # mixed precision (fp16)

# Pointcept – konwencja „eval epoch” (z issues):
#   epoch + eval_epoch mapują się na:
#     dataset_loop     = epoch / eval_epoch
#     real_train_epoch = eval_epoch
#   czyli równoważnie:
#     epoch = real_train_epoch * dataset_loop
#
# Log train ma format:
#   Train: [cur_epoch/epoch][cur_iter/iters_per_epoch]
# gdzie:
#   iters_per_epoch = len(train_loader)
#   ≈ ceil(len(train_dataset_effective) / batch_size)
#   a len(train_dataset_effective) bywa ~ len(train_dataset) * dataset_loop
#   (jeśli/jeżeli `loop` faktycznie powiela dataset / zwiększa liczbę próbek na epokę).
# epoch = 60 * 5
# eval_epoch = 60
epoch = 1000
eval_epoch = 200

DATASET_NAME = "ZAHADataset"
DATA_ROOT = "data/zaha"
GRID_SIZE = 0.2 # 20cm
TRAIN_POINTS_LIMIT = 1024 * 10 * 40

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
    "displacement", # (N, 3)
)

FEATURE_CHANNELS = sum([
    3,  # coord
    # 3,  # normal
    # 1,  # height
    # 1,  # verticality
    # 1,  # planarity
    # 1,  # linearity
    # 1,  # sphericity
    3,  # displacement
])

model = dict(
    type="DefaultSegmentorV2",
    num_classes=NUMBER_OF_CLASSES,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        in_channels=FEATURE_CHANNELS,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ZAHA",),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=IGNORE_INDEX),
        dict(
            type="LovaszLoss",
            mode="multiclass",
            loss_weight=1.0,
            ignore_index=IGNORE_INDEX,
        ),
    ],
)

# Optimizer (PTv3/Pointcept-style)
optimizer = dict(
    type="AdamW",
    lr=0.006,
    weight_decay=0.05,
    betas=(0.9, 0.999),
    eps=1e-8,
)

# Scheduler (Pointcept PTv3 semseg)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],      # [base_lr, block_lr]
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# LR dla bloków attention (≈0.1x)
param_dicts = [dict(keyword="block", lr=0.0006)]

data = dict(
    names=CLASSES,
    num_classes=NUMBER_OF_CLASSES,
    ignore_index=IGNORE_INDEX,

    train=dict(
        type=DATASET_NAME,
        split=("train",),
        data_root=DATA_ROOT,
        ignore_index=IGNORE_INDEX,
        test_mode=False,
        loop=4,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomRotate",
                angle=[-1, 1],
                axis="z",
                center=[0, 0, 0],
                p=0.5,
            ),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=GRID_SIZE,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_displacement=True,
            ),
            # dict(
            #     type="GridCrop",
            #     cell_size=8.0,
            #     point_max=TRAIN_POINTS_LIMIT,
            #     min_points_in_cell=64,
            #     p_edge=0.4, # 40% - agresywniej samplinguj brzegi
            #     edge_width=1,
            #     sample_mode="knn",
            #     shuffle=True,
            # ),
            dict(
                type="ClassAwareCropKNN",
                point_max=TRAIN_POINTS_LIMIT,
                target_classes=(3, 6, 13), # door, deco, blinds
                p_target=0.7,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "displacement",
                ),
                feat_keys=FEATURE_KEYS,
            ),
        ],
    ),

    val=dict(
        type=DATASET_NAME,
        split="validation",
        data_root=DATA_ROOT,
        ignore_index=IGNORE_INDEX,
        test_mode=False,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=GRID_SIZE,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
                return_displacement=True,
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
                    "displacement",
                ),
                feat_keys=FEATURE_KEYS,
            ),
        ],
    ),

    test=dict(
        data_root=DATA_ROOT,
        type=DATASET_NAME,
        split="test",
        ignore_index=IGNORE_INDEX,
        test_mode=True,
        transform=[
            dict(type="CenterShift", apply_z=True),
        ],
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=GRID_SIZE,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_displacement=True,
                return_inverse=True
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=(
                        "coord",
                        "grid_coord",
                        "index",
                        "displacement",
                    ),
                    feat_keys=FEATURE_KEYS,
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
