from imgaug import augmenters as iaa

def get_augmentor_1():
    # performed best on unet
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(1.0),  # horizontal flip
        iaa.Flipud(1.0),  # vertical flip
        iaa.Affine(rotate=(-15, 15)),
        iaa.Affine(shear=(-20, 20)),
        iaa.Multiply((0.5, 1.5), per_channel=0.5),  # lighten or darken
    ])
    return augmentation

def get_augmentor_2():
    # performed well on simple cnn
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),  # horizontal flip
        iaa.Flipud(0.5),  # vertical flip
        iaa.Multiply((0.5, 1.5), per_channel=0.5),  # lighten or darken
    ])
    return augmentation

def get_augmentor_3():
    # flipping only
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),  # horizontal flip
        iaa.Flipud(0.5),  # vertical flip
    ])
    return augmentation

def get_augmentor_4():
    # also performed well on simple cnn
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Flipud(1.0),  # vertical flip
        iaa.Multiply((0.5, 1.5), per_channel=0.5),  # lighten or darken
        iaa.Affine(shear=(-20, 20)),
    ])
    return augmentation