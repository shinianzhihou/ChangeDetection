norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='SiameseEfficientNet',
        name='efficientnet_b3',
        fusion='conc',
        conv1x1=False,
        attention=False,
        conv1x1_in=[48, 96, 272, 768],
        conv1x1_out=[24, 48, 136, 384],
        in_index=[0, 2, 4, 6],
        pretrained=False),
    decode_head=dict(
        type='UPerHead',
        in_channels=[48, 96, 272, 768],
        in_index=[0, 2, 4, 6],
        pool_scales=(1, 2, 3, 6),
        channels=80,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='MultiLoss',
            losses=[
                dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0,
                    class_weight=[1.0, 1.0]),
                dict(type='DiceLoss', loss_weight=1.0)
            ],
            weights=[1.0, 0.5])),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'TwoInputDataset'
img_norm_cfg = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
train_pipeline = [
    dict(type='RandomResizedCrop', height=512, width=512, p=0.7),
    dict(type='RandomRotate90', p=0.5),
    dict(type='RandomBrightnessContrast', p=0.5),
    dict(type='GaussianBlur', p=0.3),
    dict(type='HorizontalFlip', p=0.3),
    dict(type='VerticalFlip', p=0.3),
    dict(
        type='Normalize',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)),
    dict(type='ToTensorV2')
]
test_pipeline = [
    dict(
        type='Normalize',
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)),
    dict(type='ToTensorV2')
]
data_root = '/workspace/dataset/stb'
train_file = '/workspace/dataset/stb/train.lky.txt'
val_file = '/workspace/dataset/stb/val.lky.txt'
test_file = '/workspace/dataset/stb/val.lky.txt'
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        type='TwoInputDataset',
        meta_file='/workspace/dataset/stb/train.lky.txt',
        data_root='/workspace/dataset/stb',
        sep='	',
        imdecode_backend='pillow',
        c255_t1_in_mask=False,
        pipeline=[
            dict(type='RandomResizedCrop', height=512, width=512, p=0.7),
            dict(type='RandomRotate90', p=0.5),
            dict(type='RandomBrightnessContrast', p=0.5),
            dict(type='GaussianBlur', p=0.3),
            dict(type='HorizontalFlip', p=0.3),
            dict(type='VerticalFlip', p=0.3),
            dict(
                type='Normalize',
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
            dict(type='ToTensorV2')
        ]),
    val=dict(
        type='TwoInputDataset',
        meta_file='/workspace/dataset/stb/val.lky.txt',
        data_root='/workspace/dataset/stb',
        sep='	',
        imdecode_backend='pillow',
        c255_t1_in_mask=False,
        pipeline=[
            dict(
                type='Normalize',
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
            dict(type='ToTensorV2')
        ]),
    test=dict(
        type='TwoInputDataset',
        meta_file='/workspace/dataset/stb/val.lky.txt',
        data_root='/workspace/dataset/stb',
        sep='	',
        imdecode_backend='pillow',
        c255_t1_in_mask=False,
        pipeline=[
            dict(
                type='Normalize',
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
            dict(type='ToTensorV2')
        ]))
optimizer = dict(type='Adam', lr=0.001)
optimizer_config = dict()
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.1,
    min_lr_ratio=5e-05,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=10000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000000000, metric='mIoU')
log_config = dict(
    interval=100, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
work_dir = '../work_dir/cd_stb/upernet_efficientnet-b3_512x512_stb_lr1e-2_1e-4_debug_Adam_weight_noabs'
gpu_ids = range(0, 1)
