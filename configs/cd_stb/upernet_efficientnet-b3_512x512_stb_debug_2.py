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
#     backbone=dict(
#         type='SiameseEfficientNet',
#         name='efficientnet_b3',
#         fusion='conc',
#         conv1x1=False,
#         attention=False,
#         non_siamese=False,
#         conv1x1_in = [i*2 for i in [24, 48, 136, 384]],
#         conv1x1_out = [i*1 for i in [24, 48, 136, 384]],
#         in_index = [0,2,4,6],
#         pretrained=True,
#     ),
    
        backbone=dict(
        type='SiameseEfficientNet',
        name='efficientnet_b3',
        fusion='conc',
        conv1x1=False,
        attention=False,
        non_siamese=False,
        conv1x1_in = [i*2 for i in [24, 48, 136, 384]],
        conv1x1_out = [i*1 for i in [24, 48, 136, 384]],
        in_index = [0,2,4,6],
        pretrained=True,
    ),
    
    decode_head=dict(
        type='UPerHead',
        in_channels=[48, 96, 272, 768],
        in_index=[0, 2, 4, 6],
        pool_scales=(1, 2, 3, 6),
        channels=80,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
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

# dataset settings
dataset_type = 'TwoInputDataset'
img_norm_cfg = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

train_pipeline = [
    dict(type='RandomResizedCrop', height=512, width=512, scale=(1/16,2.0), ratio=(0.5,2.0), p=0.7),
    dict(type='RandomRotate90',p=0.5),
    dict(type='RandomBrightnessContrast',p=0.5),
    dict(type='GaussianBlur',p=0.3),
    dict(type='HorizontalFlip', p=0.3),
    dict(type='VerticalFlip', p=0.3),
    dict(type='Normalize',**img_norm_cfg),
    dict(type='ToTensorV2'),
]

test_pipeline = [
    dict(type='Normalize',**img_norm_cfg),
    dict(type='ToTensorV2'),
]

data_root = '/workspace/dataset/stb'
train_file = '/workspace/dataset/stb/train.3+1.2.txt'
val_file = train_file
test_file = train_file

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
#     paramwise_cfg = dict(
#         custom_keys={
#             'head': dict(lr_mult=4.)}),
    type='Adam', 
    lr=1e-3)

optimizer_config = dict()
# learning policy
# lr_config = dict(policy='poly', power=0.9, min_lr=5e-5, by_epoch=False)
# lr_config = dict(policy='step', gamma=0.1, step=[8000,14000], by_epoch=False)
# lr_config = dict(policy='step', gamma=0.1, step=[7000,9000], by_epoch=False)
lr_config = dict(policy='CosineAnnealing',
                 warmup='linear',
                 warmup_iters=1000,
                 warmup_ratio=1.0/10,
                 min_lr_ratio=5e-5,
                 by_epoch=False)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=12000)
checkpoint_config = dict(by_epoch=False, interval=500)
evaluation = dict(interval=1000000000, metric='mIoU')

# runtime
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
#         dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
seed = 318
