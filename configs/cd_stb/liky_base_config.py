_base_ = [
#     '../_base_/models/cd_vit.py',
    # '../_base_/datasets/two_input.py',
    # '../_base_/default_runtime.py',
    # '../_base_/schedules/schedule_40k.py'
]


# model settings
norm_cfg = dict(type='BN', requires_grad=True)  # TO: BN
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='SiameseEfficientNet',
        name='efficientnet_b1',
        fusion='diff',
        # pretrained=True,
        checkpoint_path='../weights/efficientnet_b1-533bc792.pth',
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[24, 40, 112, 320],
        in_index=[1, 2, 3, 4],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0, class_weight=[0.8, 1.2])),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=112,
        in_index=3,
        channels=256,
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
    dict(type='RandomResizedCrop', height=512, width=512, p=0.8),
    dict(type='RandomRotate90',p=0.5),
    dict(type='RandomBrightnessContrast',p=0.2),
    dict(type='GaussianBlur',p=0.3),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='VerticalFlip', p=0.5),
    dict(type='Normalize',**img_norm_cfg),
    dict(type='ToTensorV2'),
]

test_pipeline = [
    # dict(type='CenterCrop', height=256, width=256, p=1.0),
    dict(type='Normalize',**img_norm_cfg),
    dict(type='ToTensorV2'),
]

data_root = '/cache'
train_file = './work_dirs/cd_stb/meta_files/train.v1.txt'
val_file = './work_dirs/cd_stb/meta_files/val.v1.txt'
test_file = './work_dirs/cd_stb/meta_files/test.txt'

data = dict(
    samples_per_gpu=12,
    workers_per_gpu=4,
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
    lr=1e-3,
    momentum=0.9,
    weight_decay=0.0005)

optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=6000)
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=500000, metric='mIoU')

# runtime
# yapf:disable
log_config = dict(
    interval=50,
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
