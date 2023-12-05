import os 

# dist_params = dict(backend='nccl')
# cudnn_benchmark = True 

env_cfg = dict(dist_cfg = dict(backend='nccl'), cudnn_benchmark = True)

log_level = 'INFO' 
load_from = None 
resume_from = None 
custom_imports = dict(imports=['geospatial_fm']) 

# num_frames = 3
num_frames = 1
img_size = 224
num_workers = 2

# model
# TO BE DEFINED BY USER: model path
pretrained_weights_path = "/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data/filtered_Prithvi_100M.pt"
num_layers = 6
patch_size = 16
embed_dim = 768
num_heads = 8
tubelet_size = 1
max_epochs = 80
eval_epoch_interval = 5
output_embed_dim = embed_dim*num_frames

# loss_weights_multi = [
#     0.386375, 0.661126, 0.548184, 0.640482, 0.876862, 0.925186, 3.249462,
#     1.542289, 2.175141, 2.272419, 3.062762, 3.626097, 1.198702
# ]
loss_weights_multi = [0.05004486, 0.05469906, 0.48799205, 0.0532651, 
                      0.19849055, 0.04613963, 0.05042878, 0.05893997]

# loss_func = dict(
#     type='CrossEntropyLoss',
#     use_sigmoid=False,
#     class_weight=loss_weights_multi,
#     avg_non_ignore=True)

loss_func = dict(
    type='CrossEntropyLoss',
    use_sigmoid=True,
    class_weight=loss_weights_multi,
    )

# TO BE DEFINED BY USER: Save directory
experiment = 'results'
project_dir = 'task2_gfm'
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir

gpu_ids = range(0, 1)
# dataset_type = 'GeospatialDataset'
dataset_type = 'MultiLabelGeospatialDataset'

# TO BE DEFINED BY USER: data directory
data_root = "/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data" #updated

splits = dict(
    train = "/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data_splits/multi_label_classification/train.txt",
    val= "/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data_splits/multi_label_classification/val.txt",
    test=  "/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data_splits/multi_label_classification/val.txt"
    )

bands = [0, 1, 2, 3, 4, 5]
tile_size = img_size


img_norm_cfg = dict(
    means=[
        494.905781, 815.239594, 924.335066, 2968.881459, 2634.621962, 1739.579917
    ],
    stds=[
        284.925432, 357.84876, 575.566823, 896.601013, 951.900334, 921.407808
    ]) # needs update

# data_preprocessor = dict(
#     mean=[494.905781, 815.239594, 924.335066, 2968.881459, 2634.621962, 1739.579917],
#     std=[284.925432, 357.84876, 575.566823, 896.601013, 951.900334, 921.407808],
#     )


#orig_nsize = 512
#crop_size = (tile_size, tile_size)

# train_pipeline = [
#     dict(type='LoadGeospatialImageFromFile', to_float32=True),
#     dict(type='LoadGeospatialAnnotations', reduce_zero_label=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
#      # to channels first
#     dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
#     dict(type='TorchNormalize', **img_norm_cfg),
#     dict(type='TorchRandomCrop', crop_size=crop_size),
#     dict(type='Reshape', keys=['img'], new_shape=(len(bands), num_frames, tile_size, tile_size)),
#     dict(type='Reshape', keys=['gt_semantic_seg'], new_shape=(1, tile_size, tile_size)),
#     dict(type='CastTensor', keys=['gt_semantic_seg'], new_type="torch.LongTensor"),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg']),
# ]

train_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True, channels_last=True),
    dict(type='BandsExtract', bands=bands),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='ToTensor', keys=['img']),
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type='TorchNormalize', **img_norm_cfg),
    dict(type='Reshape', keys=['img'], new_shape=(len(bands), num_frames, tile_size, tile_size)),
    dict(type='PackInputs'),
    ]

# test_pipeline = [
#     dict(type='LoadGeospatialImageFromFile', to_float32=True),
#     dict(type='ToTensor', keys=['img']),
#      # to channels first
#     dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
#     dict(type='TorchNormalize', **img_norm_cfg),
#     dict(type='Reshape', keys=['img'], new_shape=(len(bands), num_frames, -1, -1), look_up = {'2': 1, '3': 2}),
#     dict(type='CastTensor', keys=['img'], new_type="torch.FloatTensor"),
#     dict(type='CollectTestList', keys=['img'],
#          meta_keys=['img_info', 'seg_fields', 'img_prefix', 'seg_prefix', 'filename', 'ori_filename', 'img',
#                     'img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg']),
# ]

test_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True, channels_last=True),
    dict(type='BandsExtract', bands=bands),
    dict(type='ToTensor', keys=['img']),
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type='TorchNormalize', **img_norm_cfg),
    dict(type='Reshape', keys=['img'], new_shape=(len(bands), num_frames, -1, -1), look_up = {'2': 1, '3': 2}),
    dict(type='CastTensor', keys=['img'], new_type="torch.FloatTensor"),
    dict(type='CollectTestList', keys=['img'],
         meta_keys=['img_info', 'seg_fields', 'img_prefix', 'seg_prefix', 'filename', 'ori_filename', 'img',
                    'img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg']),
    ]

# CLASSES = ('Natural Vegetation', 
#            'Forest', 
#            'Corn', 
#            'Soybeans', 
#            'Wetlands', 
#            'Developed/Barren', 
#            'Open Water', 
#            'Winter Wheat', 
#            'Alfalfa', 
#            'Fallow/Idle Cropland', 
#            'Cotton', 
#            'Sorghum', 
#            'Other')

CLASSES = ("Built Area",
            "Grass",
            "Flooded Vegetation",
            "Crops",
            "Shrub & Scrub",
            "Trees",
            "Water",
            "Bare Ground")

# dataset = 'GeospatialDataset'
dataset = 'MultiLabelGeospatialDataset'

# data = dict(
#     samples_per_gpu=8,
#     workers_per_gpu=4,
#     train=dict(
#         type=dataset,
#         CLASSES=CLASSES,
#         reduce_zero_label=True,
#         data_root=data_root,
#         img_dir='training_chips',
#         ann_dir='training_chips',
#         pipeline=train_pipeline,
#         img_suffix='_merged.tif',
#         seg_map_suffix='.mask.tif',
#         split=splits['train']),
#     val=dict(
#         type=dataset,
#         CLASSES=CLASSES,
#         reduce_zero_label=True,
#         data_root=data_root,
#         img_dir='validation_chips',
#         ann_dir='validation_chips',
#         pipeline=test_pipeline,
#         img_suffix='_merged.tif',
#         seg_map_suffix='.mask.tif',
#         split=splits['val']
#     ),
#     test=dict(
#         type=dataset,
#         CLASSES=CLASSES,
#         reduce_zero_label=True,
#         data_root=data_root,
#         img_dir='validation_chips',
#         ann_dir='validation_chips',
#         pipeline=test_pipeline,
#         img_suffix='_merged.tif',
#         seg_map_suffix='.mask.tif',
#         split=splits['val']
#     ))

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type=dataset,
        classes=CLASSES,
        data_root=data_root,
        ann_file="/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data/annotations/dataset_dict.json",
        pipeline=train_pipeline,
        split=splits['train']))

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=None,
    dataset=dict(
        type=dataset,
        classes=CLASSES,
        data_root=data_root,
        ann_file="/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data/annotations/dataset_dict.json",
        pipeline=train_pipeline,
        split=splits['val']))

test_dataloader = val_dataloader

# optimizer = dict(
#     type='Adam', lr=1.5e-05, betas=(0.9, 0.999), weight_decay=0.05)
# optimizer_config = dict(grad_clip=None)

optim_wrapper=dict(
    optimizer = dict(type='Adam', 
                     lr=1.5e-05, 
                     betas=(0.9, 0.999), 
                     weight_decay=0.05),
    clip_grad=None,
    )

# lr_config = dict(
#     policy='poly',
#     warmup='linear',
#     warmup_iters=1500,
#     warmup_ratio=1e-06,
#     power=1.0,
#     min_lr=0.0,
#     by_epoch=False)

param_scheduler = [
    dict(type='PolyLR',
         power=1.0,
         begin=0,
         end=1e06, 
         eta_min=0.0,
         by_epoch=False),
    ]

# log_config = dict(
#     interval=10,
#     hooks=[dict(type='TextLoggerHook'),
#            dict(type='TensorboardLoggerHook')])

# checkpoint_config = dict(
#     by_epoch=True,
#     interval=100,
#     out_dir=save_path)

# workflow = [('train', 1)]

default_hooks=dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint = dict(type='CheckpointHook', by_epoch=True, interval=100, out_dir=save_path),
    visualization=dict(type='VisualizationHook', enable=True),
    )

visualizer = dict(
    type='UniversalVisualizer', vis_backends=[dict(type='LocalVisBackend'),
                                              dict(type='TensorboardVisBackend'),])

# evaluation = dict(interval=eval_epoch_interval, metric='mIoU', pre_eval=True, save_best='mIoU', by_epoch=True) # needs change
# reduce_train_set = dict(reduce_train_set=False)
# reduce_factor = dict(reduce_factor=1)

val_evaluator = [
  dict(type='AveragePrecision'),
  dict(type='MultiLabelMetric', average='macro'),
  dict(type='MultiLabelMetric', average='micro'),
  ]
test_evaluator = val_evaluator


# runner = dict(type='EpochBasedRunner', max_epochs=max_epochs, val_interval=1)
# train_cfg=dict()
# test_cfg=dict(mode='slide', stride=(int(img_size/2), int(img_size/2)), crop_size=(img_size, img_size))

train_cfg=dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict()
test_cfg=dict()


norm_cfg = dict(type='BN', requires_grad=True)

# model = dict(
#     type='TemporalEncoderDecoder',
#     frozen_backbone=False,
#     backbone=dict(
#         type='TemporalViTEncoder',
#         pretrained=pretrained_weights_path,
#         img_size=img_size,
#         patch_size=patch_size,
#         num_frames=num_frames,
#         tubelet_size=1,
#         in_chans=len(bands),
#         embed_dim=embed_dim,
#         depth=6,
#         num_heads=num_heads,
#         mlp_ratio=4.0,
#         norm_pix_loss=False),
#     neck=dict(
#         type='ConvTransformerTokensToEmbeddingNeck',
#         embed_dim=embed_dim*num_frames,
#         output_embed_dim=output_embed_dim,
#         drop_cls_token=True,
#         Hp=14,
#         Wp=14),
#     decode_head=dict(
#         num_classes=len(CLASSES),
#         in_channels=output_embed_dim,
#         type='FCNHead',
#         in_index=-1,
#         channels=256,
#         num_convs=1,
#         concat_input=False,
#         dropout_ratio=0.1,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         align_corners=False,
#         loss_decode=loss_func),
#     auxiliary_head=dict(
#         num_classes=len(CLASSES),
#         in_channels=output_embed_dim,
#         type='FCNHead',
#         in_index=-1,
#         channels=256,
#         num_convs=2,
#         concat_input=False,
#         dropout_ratio=0.1,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         align_corners=False,
#         loss_decode=loss_func),
#     train_cfg=dict(),
#     test_cfg=dict(mode='slide', stride=(int(img_size/2), int(img_size/2)), crop_size=(img_size, img_size))) # needs change

model = dict(
    type='GeospatialMultiLabelClassifier',
    frozen_backbone=False,
    #data_preprocessor=data_preprocessor,
    backbone=dict(
        type='TemporalViTEncoder',
        pretrained=pretrained_weights_path,
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=1,
        in_chans=len(bands),
        embed_dim=embed_dim,
        depth=6,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_pix_loss=False),
    cls_head=dict(
        type='MultiLabelClsHead',
        loss=loss_func),
    # head=dict(
    #     type='MultiLabelLinearClsHead',
    #     num_classes=len(CLASSES),
    #     in_channels=output_embed_dim,
    #     loss=loss_func),
    )

auto_resume = False
