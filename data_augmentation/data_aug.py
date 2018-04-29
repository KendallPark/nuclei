from imgaug import augmenters as iaa

def get_augmentor():
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),  # horizontal flip
        iaa.Flipud(0.5),  # vertical flip
        iaa.Affine(translate_px={"x": (-15, 15), "y": (-15, 15)}),
        iaa.Affine(rotate=(-45, 45)),
        iaa.Affine(shear=(-20, 20)),
        iaa.Multiply((0.5, 1.5), per_channel=0.5),  # lighten or darken
    ])
    return augmentation
