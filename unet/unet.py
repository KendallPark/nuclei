import os
import sys
import random
import warnings
import skimage.io
import numpy as np
import pandas as pd
import cv2
#import matplotlib.pyplot as plt
import skimage.segmentation
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
import tensorflow as tf
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from IPython import embed

ROOT_DIR = os.path.abspath("./")
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
from data.extractor_id import extraction

# params
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
TRAIN_PATH = 'data/stage1_train/'
TEST_PATH = 'data/stage1_test/'

test_csv = pd.read_csv('data/stage1_solution.csv')
test_ids = next(os.walk(TEST_PATH))[1]

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

num_of_epoch = 100
filename = 'model-no-augmentation.h5'

sizes_test = []

def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = list(map(int, rle.split()))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    rle -= 1
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[1], shape[0]]).T
    return mask

def load_resize_test_data():
    X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_test = []
    # Y_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = TEST_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_test[n] = img
        mask = np.zeros((sizes_test[-1][0], sizes_test[-1][1], 1), dtype=np.bool)
        # embed()
        for thingy in test_csv[test_csv['ImageId'] == id_].values:
            mask_file, encoding, height, width, whatever = thingy
            mask_ = rle_decode(encoding, [height, width]) # imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (sizes_test[-1][0], sizes_test[-1][1]), mode='constant',
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_test.append(mask[:, :, 0])

    print('Done!')
    return X_test, Y_test

def load_resize_data(train_ids, validation_ids):
    # Get train and test IDs

    # Get and resize train images and masks
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing train images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_train[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_train[n] = mask

    X_val = np.zeros((len(validation_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_val = np.zeros((len(validation_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    print('Getting and resizing validation images and masks ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(validation_ids), total=len(validation_ids)):
        path = TRAIN_PATH + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        X_val[n] = img
        mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                          preserve_range=True), axis=-1)
            mask = np.maximum(mask, mask_)
        Y_val[n] = mask

    return X_train, Y_train, X_val, Y_val


def get_data():
    validation_ids = extraction(0.1, "data/stage1_train_classses.csv", "data/stage1_train")
    all_ids = next(os.walk(TRAIN_PATH))[1]
    train_ids = [value for value in all_ids if value not in validation_ids]

    print("## Train IDs")
    print(train_ids)
    print("## Validation IDs")
    print(validation_ids)
    return load_resize_data(train_ids, validation_ids)


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def mean_iou_test(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(tf.convert_to_tensor(y_true), tf.convert_to_tensor(y_pred_), 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    # return prec
    mean = K.mean(K.stack(prec), axis=0)
    return mean



def build_unet():
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255)(inputs)

    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(s)
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    c9 = Conv2D(2, 3, activation='relu', padding='same')(c9)
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])
    model.summary()

    return model


def train(X_train, Y_train, X_val, Y_val, model, filename):
    earlystopper = EarlyStopping(patience=5, verbose=1)
    checkpointer = ModelCheckpoint(filename, verbose=1, save_best_only=True)
    results = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), batch_size=8, epochs=150,
                        callbacks=[checkpointer])


def main():
    if sys.argv[1] == "test":
        model = load_model(os.path.join("demo", "models", "unet", "unet_aug_0.h5"), custom_objects={'mean_iou': mean_iou})
        X_test, Y_test = load_resize_test_data()
        preds_test = model.predict(X_test, verbose=1)
        preds_test_t = (preds_test > 0.5).astype(np.uint8)

        # Create list of upsampled test masks
        preds_test_upsampled = []
        for i in range(len(preds_test)):
            preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                               (sizes_test[i][0], sizes_test[i][1]),
                                               mode='constant', preserve_range=True))
        means = []
        for i in range(len(preds_test_upsampled)):
            prediction = preds_test_upsampled[i]
            actual = Y_test[i]
            # embed()
            means.append(mean_iou_test(actual, prediction))
        for mean in means:
            sess = tf.Session()
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            # sess.run(up_opt)
            res = sess.run(mean)
            print(res)
            # return res
        # print(mean)

    else:
        model = build_unet()
        [X_train, Y_train, X_val, Y_val] = get_data()
        train(X_train, Y_train, X_val, Y_val, model, filename)



if __name__ == '__main__':
    main()
