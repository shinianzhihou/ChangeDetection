_base_ = [
    '../_base_/models/cd_vit.py',
    # '../_base_/datasets/two_input.py', 
    # '../_base_/default_runtime.py',
    # '../_base_/schedules/schedule_40k.py'
]


# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='CDVitV2',
        backbone_choice='resnet18',
        num_images=2,
        image_size=256,
        feature_size=64,
        patch_size=4,
        in_channels=128,
        out_channels=32,
        encoder_dim=512,
        encoder_heads=8,
        encoder_dim_heads=64,
        encoder_depth=4,
        attn_dropout=0.1,
        ff_dropout=0.1),
    decode_head=dict(
        type='CDVitHead',
        in_channels=64,
        in_index=0,
        channels=32,
        num_convs=2,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
dataset_type = 'TwoInputDataset'
img_norm_cfg = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

train_pipeline = [
    dict(type='RandomResizedCrop', height=256, width=256, p=0.5),
    dict(type='RandomRotate90',p=1),
    dict(type='RandomBrightnessContrast',p=0.2),
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

data_root = ''
train_file = '/path/to/Building-CD/dataset-224/train.txt'
val_file = '/path/to/Building-CD/dataset-224/val.txt'
test_file = '/path/to/Building-CD/dataset-224/val.txt'

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        meta_file=train_file,
        data_root=data_root,
        sep='\t',
        imdecode_backend='cv2',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        meta_file=val_file,
        data_root=data_root,
        sep='\t',
        imdecode_backend='cv2',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        meta_file=val_file,
        data_root=data_root,
        sep='\t',
        imdecode_backend='cv2',
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=400000000, metric='mIoU')

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