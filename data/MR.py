_base_ = ['blocks']
# model settings
model = dict(
    strides=(1, 2, 4, 8),
    buffer_size=1024,
    max_num_moment=50,
    pyramid_cfg=dict(type="ConvPyramid"),
    pooling_cfg=dict(type="AdaPooling"),
    class_head_cfg=dict(type="ConvHead", kernal_size=3),
    coord_head_cfg=dict(type="ConvHead", kernal_size=3),
    loss_cfg=dict(
            type='BundleLoss',
            sample_radius=1.5,
            loss_cls=dict(type='FocalLoss'),
            loss_reg=dict(type='L1Loss'),
            loss_sal=dict(type='SampledNCELoss'),
        ),
)