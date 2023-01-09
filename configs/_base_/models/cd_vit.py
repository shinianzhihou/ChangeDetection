norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='CDVit',
        backbone_choice='resnet18',
        num_images=2,
        image_size=224,
        feature_size=28,
        patch_size=4,
        in_channels=128,
        out_channels=32,
        encoder_dim=512,
        encoder_heads=8,
        encoder_dim_heads=64,
        encoder_depth=4,
        attn_dropout=0.0,
        ff_dropout=0.0),
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
