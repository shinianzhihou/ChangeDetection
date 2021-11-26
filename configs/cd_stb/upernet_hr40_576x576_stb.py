_base_ = [
#     '../_base_/models/cd_vit.py',
    # '../_base_/datasets/two_input.py', 
    # '../_base_/default_runtime.py',
    # '../_base_/schedules/schedule_40k.py'
]


# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='/workspace/models/pretrain_models/hrnetv2_w40_imagenet_pretrained.pth', # change it to /path/to/hrnetv2_w40_imagenet_pretrained.pth
    backbone=dict(
        type='TwoStreamHRNet',
        norm_cfg=norm_cfg,
        norm_eval=False,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(40, 80)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(40, 80, 160)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(40, 80, 160, 320)))),
    decode_head=dict(
        type='UPerHead',
        in_channels=[40, 80, 160, 320],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=80,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            class_weight=[0.8,1.2])),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=160,
        in_index=2,
        channels=40,
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
    dict(type='RandomResizedCrop', height=576, width=576, p=1.0),
    dict(type='RandomRotate90',p=0.8),
    dict(type='RandomBrightnessContrast',p=0.2),
    dict(type='GaussianBlur',p=0.3),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='VerticalFlip', p=0.5),
    dict(type='Normalize',**img_norm_cfg),
    dict(type='ToTensorV2'),
]

test_pipeline = [
    dict(type='CenterCrop', height=512, width=512, p=1.0),
    dict(type='Normalize',**img_norm_cfg),
    dict(type='ToTensorV2'),
]

data_root = '/workspace/dataset/stb'  # change it to /path/to/data_root
train_file = '/workspace/dataset/stb/train.6.txt' # change it to /path/to/train.6.txt
val_file = '/workspace/dataset/stb/test.txt' # change it to /path/to/test.txt
test_file = '/workspace/dataset/stb/test.txt' # change it to /path/to/test.txt

data = dict(
    samples_per_gpu=3,
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
    lr=0.01, 
    momentum=0.9, 
    weight_decay=0.0005)

optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-5, by_epoch=False)
# lr_config = dict(policy='CosineAnnealing',
#                  warmup='linear',
#                  warmup_iters=1000,
#                  warmup_ratio=1.0/10,
#                  min_lr_ratio=1e-5,
#                  by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=180000)
checkpoint_config = dict(by_epoch=False, interval=10000)
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
