# dataset settings
dataset_type = 'TwoInputDataset'
img_norm_cfg = dict(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

train_pipeline = [
    dict(type='RandomResizedCrop', height=224, width=224, p=1.0),
    dict(type='HorizontalFlip', p=0.5),
    dict(type='Normalize',**img_norm_cfg),
    dict(type='ToTensorV2'),
]

test_pipeline = [
    dict(type='CenterCrop', height=224, width=224, p=1.0),
    dict(type='Normalize',**img_norm_cfg),
    dict(type='ToTensorV2'),
]

data_root = ''
train_file = ''
val_file = ''
test_file = ''

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
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
