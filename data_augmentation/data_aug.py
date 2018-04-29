from imgaug import augmenters as iaa

def get_augmentor():
    # iaa.seed(1)
    # augmentation = iaa.SomeOf((0, 2), [
    #     iaa.Fliplr(0.5),
    #     iaa.Flipud(0.5),
    #     iaa.OneOf([iaa.Affine(rotate=90),
    #                 iaa.Affine(rotate=180),
    #                 iaa.Affine(rotate=270)]),
    #     iaa.Multiply((0.8, 1.5)),
    #     iaa.GaussianBlur(sigma=(0.0, 5.0))
    # ])
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),  # horizontal flip
        iaa.Flipud(0.5),  # vertical flip
        iaa.Affine(translate_px={"x": (-15, 15), "y": (-15, 15)}),
        iaa.Affine(rotate=(-45, 45)),
        iaa.Affine(shear=(-20, 20)),
        iaa.Multiply((0.5, 1.5), per_channel=0.5),  # lighten or darken
    ])
    return augmentation
