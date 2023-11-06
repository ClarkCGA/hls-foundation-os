CLASSES = (
    'Built Area',
    'Grass',
    'Flooded Vegetation',
    'Crops',
    'Shrub & Scrub',
    'Trees',
    'Water',
    'Bare Ground',
)
auto_resume = False
bands = [
    0,
    1,
    2,
    3,
    4,
    5,
]
checkpoint_config = dict(
    by_epoch=True, interval=100, out_dir='task2_gfm/results')
cudnn_benchmark = True
custom_imports = dict(imports=[
    'geospatial_fm',
])
data = dict(
    samples_per_gpu=4,
    test=dict(
        CLASSES=(
            'Built Area',
            'Grass',
            'Flooded Vegetation',
            'Crops',
            'Shrub & Scrub',
            'Trees',
            'Water',
            'Bare Ground',
        ),
        ann_dir='validation_chips',
        data_root=
        '/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data',
        img_dir='validation_chips',
        pipeline=[
            dict(
                channels_last=True,
                to_float32=True,
                type='LoadGeospatialImageFromFile'),
            dict(keys=[
                'img',
            ], type='ToTensor'),
            dict(keys=[
                'img',
            ], order=(
                2,
                0,
                1,
            ), type='TorchPermute'),
            dict(
                means=[
                    494.905781,
                    815.239594,
                    924.335066,
                    2968.881459,
                    2634.621962,
                    1739.579917,
                ],
                stds=[
                    284.925432,
                    357.84876,
                    575.566823,
                    896.601013,
                    951.900334,
                    921.407808,
                ],
                type='TorchNormalize'),
            dict(
                keys=[
                    'img',
                ],
                look_up=dict({
                    '2': 1,
                    '3': 2
                }),
                new_shape=(
                    6,
                    1,
                    -1,
                    -1,
                ),
                type='Reshape'),
            dict(
                keys=[
                    'img',
                ],
                new_type='torch.FloatTensor',
                type='CastTensor'),
            dict(
                keys=[
                    'img',
                ],
                meta_keys=[
                    'img_info',
                    'seg_fields',
                    'img_prefix',
                    'seg_prefix',
                    'filename',
                    'ori_filename',
                    'img',
                    'img_shape',
                    'ori_shape',
                    'pad_shape',
                    'scale_factor',
                    'img_norm_cfg',
                ],
                type='CollectTestList'),
        ],
        reduce_zero_label=True,
        split=
        '/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data_splits/multi_label_classification/val.txt',
        type='MultiLabelGeospatialDataset'),
    train=dict(
        CLASSES=(
            'Built Area',
            'Grass',
            'Flooded Vegetation',
            'Crops',
            'Shrub & Scrub',
            'Trees',
            'Water',
            'Bare Ground',
        ),
        ann_dir='annotations',
        data_root=
        '/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data',
        img_dir='training_chips',
        pipeline=[
            dict(
                channels_last=True,
                to_float32=True,
                type='LoadGeospatialImageFromFile'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(keys=[
                'img',
            ], type='ToTensor'),
            dict(keys=[
                'img',
            ], order=(
                2,
                0,
                1,
            ), type='TorchPermute'),
            dict(
                means=[
                    494.905781,
                    815.239594,
                    924.335066,
                    2968.881459,
                    2634.621962,
                    1739.579917,
                ],
                stds=[
                    284.925432,
                    357.84876,
                    575.566823,
                    896.601013,
                    951.900334,
                    921.407808,
                ],
                type='TorchNormalize'),
            dict(keys=[
                'img',
            ], new_shape=(
                6,
                1,
                224,
                224,
            ), type='Reshape'),
            dict(type='PackInputs'),
        ],
        split=
        '/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data_splits/multi_label_classification/train.txt',
        type='MultiLabelGeospatialDataset'),
    val=dict(
        CLASSES=(
            'Built Area',
            'Grass',
            'Flooded Vegetation',
            'Crops',
            'Shrub & Scrub',
            'Trees',
            'Water',
            'Bare Ground',
        ),
        ann_dir='validation_chips',
        data_root=
        '/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data',
        img_dir='validation_chips',
        pipeline=[
            dict(
                channels_last=True,
                to_float32=True,
                type='LoadGeospatialImageFromFile'),
            dict(keys=[
                'img',
            ], type='ToTensor'),
            dict(keys=[
                'img',
            ], order=(
                2,
                0,
                1,
            ), type='TorchPermute'),
            dict(
                means=[
                    494.905781,
                    815.239594,
                    924.335066,
                    2968.881459,
                    2634.621962,
                    1739.579917,
                ],
                stds=[
                    284.925432,
                    357.84876,
                    575.566823,
                    896.601013,
                    951.900334,
                    921.407808,
                ],
                type='TorchNormalize'),
            dict(
                keys=[
                    'img',
                ],
                look_up=dict({
                    '2': 1,
                    '3': 2
                }),
                new_shape=(
                    6,
                    1,
                    -1,
                    -1,
                ),
                type='Reshape'),
            dict(
                keys=[
                    'img',
                ],
                new_type='torch.FloatTensor',
                type='CastTensor'),
            dict(
                keys=[
                    'img',
                ],
                meta_keys=[
                    'img_info',
                    'seg_fields',
                    'img_prefix',
                    'seg_prefix',
                    'filename',
                    'ori_filename',
                    'img',
                    'img_shape',
                    'ori_shape',
                    'pad_shape',
                    'scale_factor',
                    'img_norm_cfg',
                ],
                type='CollectTestList'),
        ],
        split=
        '/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data_splits/multi_label_classification/val.txt',
        type='MultiLabelGeospatialDataset'),
    workers_per_gpu=2)
data_root = '/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data'
dataset = 'MultiLabelGeospatialDataset'
dataset_type = 'MultiLabelGeospatialDataset'
dist_params = dict(backend='nccl')
embed_dim = 768
eval_epoch_interval = 5
experiment = 'results'
gpu_ids = range(0, 1)
img_norm_cfg = dict(
    means=[
        494.905781,
        815.239594,
        924.335066,
        2968.881459,
        2634.621962,
        1739.579917,
    ],
    stds=[
        284.925432,
        357.84876,
        575.566823,
        896.601013,
        951.900334,
        921.407808,
    ])
img_size = 224
launcher = 'pytorch'
load_from = None
log_config = dict(
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ],
    interval=10)
log_level = 'INFO'
loss_func = dict(
    avg_non_ignore=True,
    class_weight=[
        0.05004486,
        0.05469906,
        0.48799205,
        0.0532651,
        0.19849055,
        0.04613963,
        0.05042878,
        0.05893997,
    ],
    type='CrossEntropyLoss',
    use_sigmoid=True)
loss_weights_multi = [
    0.05004486,
    0.05469906,
    0.48799205,
    0.0532651,
    0.19849055,
    0.04613963,
    0.05042878,
    0.05893997,
]
lr_config = dict(
    by_epoch=False,
    min_lr=0.0,
    policy='poly',
    power=1.0,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06)
max_epochs = 80
model = dict(
    backbone=dict(
        depth=6,
        embed_dim=768,
        img_size=224,
        in_chans=6,
        mlp_ratio=4.0,
        norm_pix_loss=False,
        num_frames=1,
        num_heads=8,
        patch_size=16,
        pretrained=
        '/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data/Prithvi_100M.pt',
        tubelet_size=1,
        type='TemporalViTEncoder'),
    cls_head=dict(
        in_features=768,
        loss=dict(
            avg_non_ignore=True,
            class_weight=[
                0.05004486,
                0.05469906,
                0.48799205,
                0.0532651,
                0.19849055,
                0.04613963,
                0.05042878,
                0.05893997,
            ],
            type='CrossEntropyLoss',
            use_sigmoid=True),
        num_classes=8,
        type='MultiLabelClsHead'),
    frozen_backbone=False,
    type='TemporalMultiLabelClassifier')
norm_cfg = dict(requires_grad=True, type='BN')
num_frames = 1
num_heads = 8
num_layers = 6
num_workers = 2
optimizer = dict(
    betas=(
        0.9,
        0.999,
    ), lr=1.5e-05, type='Adam', weight_decay=0.05)
optimizer_config = dict(grad_clip=None)
output_embed_dim = 768
patch_size = 16
pretrained_weights_path = '/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data/Prithvi_100M.pt'
project_dir = 'task2_gfm'
resume_from = None
runner = dict(max_epochs=80, type='EpochBasedRunner')
save_path = 'task2_gfm/results'
splits = dict(
    test=
    '/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data_splits/multi_label_classification/val.txt',
    train=
    '/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data_splits/multi_label_classification/train.txt',
    val=
    '/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data_splits/multi_label_classification/val.txt'
)
test_pipeline = [
    dict(
        channels_last=True,
        to_float32=True,
        type='LoadGeospatialImageFromFile'),
    dict(keys=[
        'img',
    ], type='ToTensor'),
    dict(keys=[
        'img',
    ], order=(
        2,
        0,
        1,
    ), type='TorchPermute'),
    dict(
        means=[
            494.905781,
            815.239594,
            924.335066,
            2968.881459,
            2634.621962,
            1739.579917,
        ],
        stds=[
            284.925432,
            357.84876,
            575.566823,
            896.601013,
            951.900334,
            921.407808,
        ],
        type='TorchNormalize'),
    dict(
        keys=[
            'img',
        ],
        look_up=dict({
            '2': 1,
            '3': 2
        }),
        new_shape=(
            6,
            1,
            -1,
            -1,
        ),
        type='Reshape'),
    dict(keys=[
        'img',
    ], new_type='torch.FloatTensor', type='CastTensor'),
    dict(
        keys=[
            'img',
        ],
        meta_keys=[
            'img_info',
            'seg_fields',
            'img_prefix',
            'seg_prefix',
            'filename',
            'ori_filename',
            'img',
            'img_shape',
            'ori_shape',
            'pad_shape',
            'scale_factor',
            'img_norm_cfg',
        ],
        type='CollectTestList'),
]
tile_size = 224
train_pipeline = [
    dict(
        channels_last=True,
        to_float32=True,
        type='LoadGeospatialImageFromFile'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(keys=[
        'img',
    ], type='ToTensor'),
    dict(keys=[
        'img',
    ], order=(
        2,
        0,
        1,
    ), type='TorchPermute'),
    dict(
        means=[
            494.905781,
            815.239594,
            924.335066,
            2968.881459,
            2634.621962,
            1739.579917,
        ],
        stds=[
            284.925432,
            357.84876,
            575.566823,
            896.601013,
            951.900334,
            921.407808,
        ],
        type='TorchNormalize'),
    dict(keys=[
        'img',
    ], new_shape=(
        6,
        1,
        224,
        224,
    ), type='Reshape'),
    dict(type='PackInputs'),
]
tubelet_size = 1
work_dir = 'task2_gfm/results'
workflow = [
    (
        'train',
        1,
    ),
]
