import os 
#env_cfg = dict()
dist_params = dict(backend='nccl')  # set distributed parameters
log_level = 'INFO' # logger parameter
load_from = None # load from which checkpoint
resume_from = None # resume from which checkpoint
cudnn_benchmark = True # whether to enable cudnn benchmark
custom_imports = dict(imports=['geospatial_fm']) # import your own modules.
num_frames = 1 # changed to 1 
img_size = 224
num_workers = 2

# model
# TO BE DEFINED BY USER: model path
pretrained_weights_path = "/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data/Prithvi_100M.pt"
num_layers = 6
patch_size = 16
embed_dim = 768
num_heads = 8
tubelet_size = 1
max_epochs = 80
eval_epoch_interval = 5

loss_weights_multi = [
    0.05004486, 0.05469906, 0.48799205, 0.0532651, 0.19849055, 0.04613963, 0.05042878, 0.05893997
] # updated
loss_func = dict(
    type='CrossEntropyLoss',
    use_sigmoid=True, # changed to 'True'
    class_weight=loss_weights_multi,
    avg_non_ignore=True)
output_embed_dim = embed_dim*num_frames


# TO BE DEFINED BY USER: Save directory
experiment = 'results'
project_dir = 'task2_gfm'
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir


gpu_ids = range(0, 1)
dataset_type = 'MultiLabelGeospatialDataset' # changed to 'MultiLabelGeospatialDataset'

# TO BE DEFINED BY USER: data directory
data_root = "/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data" #updated

splits = dict(
    train = "/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data_splits/multi_label_classification/train.txt",
    val= "/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data_splits/multi_label_classification/val.txt",
    test=  "/mnt/c/My_documents/summer_project/task2_gfm/hls-foundation-os/data_splits/multi_label_classification/val.txt"
) # updated


img_norm_cfg = dict(
    means=[
        494.905781, 815.239594, 924.335066, 2968.881459, 2634.621962, 1739.579917
    ],
    stds=[
        284.925432, 357.84876, 575.566823, 896.601013, 951.900334, 921.407808
    ]) # needs update

bands = [0, 1, 2, 3, 4, 5]
tile_size = img_size

#orig_nsize = 512
#crop_size = (tile_size, tile_size)
train_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True, channels_last=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='ToTensor', keys=['img']),
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type='TorchNormalize', **img_norm_cfg),
    dict(type='Reshape', keys=['img'], new_shape=(len(bands), num_frames, tile_size, tile_size)),
    dict(type='PackInputs'),
]


# train_pipeline = [
#     dict(type='LoadGeospatialImageFromFile', to_float32=True, channels_last=True),
#     dict(type='LoadGeospatialAnnotations', reduce_zero_label=True),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='ToTensor', keys=['img', 'gt_semantic_seg']), 
#      # to channels first
#     dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
#     dict(type='TorchNormalize', **img_norm_cfg),
#     dict(type='TorchRandomCrop', crop_size=crop_size),
#     dict(type='Reshape', keys=['img'], new_shape=(len(bands), num_frames, img_size, img_size)),
#     dict(type='Reshape', keys=['gt_semantic_seg'], new_shape=(1, img_size, img_size)),
#     dict(type='CastTensor', keys=['gt_semantic_seg'], new_type="torch.LongTensor"),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg']), # needs correction
# ]

test_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True, channels_last=True),
    dict(type='ToTensor', keys=['img']),
     # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type='TorchNormalize', **img_norm_cfg),
    dict(type='Reshape', keys=['img'], new_shape=(len(bands), num_frames, -1, -1), look_up = {'2': 1, '3': 2}),
    dict(type='CastTensor', keys=['img'], new_type="torch.FloatTensor"),
    dict(type='CollectTestList', keys=['img'],
         meta_keys=['img_info', 'seg_fields', 'img_prefix', 'seg_prefix', 'filename', 'ori_filename', 'img',
                    'img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg']),
]

CLASSES = ("Built Area",
            "Grass",
            "Flooded Vegetation",
            "Crops",
            "Shrub & Scrub",
            "Trees",
            "Water",
            "Bare Ground") # already updated

dataset = 'MultiLabelGeospatialDataset' # need to get changed to 'MultiLabelGeospatialDataset'

data = dict(
    samples_per_gpu=4, # orig: 8
    workers_per_gpu=2, # orig: 4
    train=dict(
        type=dataset,
        CLASSES=CLASSES,
        #reduce_zero_label=True,
        data_root=data_root,
        img_dir='training_chips',
        ann_dir='annotations',
        pipeline=train_pipeline,
        #img_suffix='_merged.tif',
        #seg_map_suffix='.mask.tif',
        split=splits['train']),
    val=dict(
        type=dataset,
        CLASSES=CLASSES,
        #reduce_zero_label=True,
        data_root=data_root,
        img_dir='validation_chips',
        ann_dir='validation_chips',
        pipeline=test_pipeline,
        #img_suffix='_merged.tif',
        #seg_map_suffix='.mask.tif',
        split=splits['val']
    ),
    test=dict(
        type=dataset,
        CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir='validation_chips',
        ann_dir='validation_chips',
        pipeline=test_pipeline,
        #img_suffix='_merged.tif',
        #seg_map_suffix='.mask.tif',
        split=splits['val']
    )) # needs change

optimizer = dict(
    type='Adam', lr=1.5e-05, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

checkpoint_config = dict(
    by_epoch=True,
    interval=100,
    out_dir=save_path)

# evaluation = dict(interval=eval_epoch_interval, metric='mIoU', pre_eval=True, save_best='mIoU', by_epoch=True) # needs change
# reduce_train_set = dict(reduce_train_set=False)
# reduce_factor = dict(reduce_factor=1)
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
workflow = [('train', 1)]
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='TemporalMultiLabelClassifier',
    frozen_backbone=False,
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
        num_classes=len(CLASSES),
        in_features=output_embed_dim,
        loss=loss_func))

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
auto_resume = False
