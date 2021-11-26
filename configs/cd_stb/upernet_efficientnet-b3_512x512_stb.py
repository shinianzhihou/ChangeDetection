_base_ = [
#     '../_base_/models/cd_vit.py',
    # '../_base_/datasets/two_input.py', 
    # '../_base_/default_runtime.py',
    # '../_base_/schedules/schedule_40k.py'
]


# model settings
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
#     pretrained='',
    backbone=dict(
        type='SiameseEfficientNet',
        name='tf_efficientnetv2_b3',
        fusion='diff',
        pretrained=True,
#         checkpoint_path='/workspace/models/pretrain_models/tf_efficientnetv2_b0-c7cc451f.pth',
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[16, 40, 56, 112, 136, 232],
        in_index=[0, 1, 2, 3, 4, 5],
        pool_scales=(1, 2, 3, 6),
        channels=80,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[0.8, 1.2])),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=232,
        in_index=5,
        channels=20,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
dataset_type = 'TwoInputDataset'
img_norm_cfg = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

train_pipeline = [
    dict(type='RandomResizedCrop', height=512, width=512, p=1.0),
    dict(type='RandomRotate90',p=0.8),
    dict(type='RandomBrightnessContrast',p=0.2),
    dict(type='GaussianBlur',p=0.3),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='VerticalFlip', p=0.5),
    dict(type='Normalize',**img_norm_cfg),
    dict(type='ToTensorV2'),
]

test_pipeline = [
    dict(type='CenterCrop', height=256, width=256, p=1.0),
    dict(type='Normalize',**img_norm_cfg),
    dict(type='ToTensorV2'),
]

data_root = '/workspace/dataset/stb'
train_file = '/workspace/dataset/stb/train.v1.6.txt'
val_file = '/workspace/dataset/stb/test.txt'
test_file = '/workspace/dataset/stb/test.txt'

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        meta_file=train_file,                      
        data_root=data_root,
        sep='\t',
        imdecode_backend='pillow',
        c255_t1_in_mask=False,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        meta_file=val_file,
        data_root=data_root,
        sep='\t',
        imdecode_backend='pillow',
        c255_t1_in_mask=False,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        meta_file=val_file,
        data_root=data_root,
        sep='\t',
        imdecode_backend='pillow',
        c255_t1_in_mask=False,
        pipeline=test_pipeline))

# optimizer
optimizer = dict(
    paramwise_cfg = dict(
        custom_keys={
            'head': dict(lr_mult=4.)}),
    type='SGD', 
    lr=0.05, 
    momentum=0.9, 
    weight_decay=0.0005)

optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=5000)
evaluation = dict(interval=400000000, metric='mIoU')

# runtime
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
