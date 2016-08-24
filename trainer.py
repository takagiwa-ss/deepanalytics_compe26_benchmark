#! /usr/bin/python
# -*- coding: utf-8 -*-

#
# Copyright© 2016 Sansan, Inc. All Rights Reserved.
#

import logging
import os
import numpy as np
import keras
import pandas as pd
from PIL import Image

# 予測出力ファイル
fn_prediction = 'prediction.csv'

# ダウンロードデータ
train_csv    = 'train.csv'
train_images = 'train_images'
test_csv     = 'test.csv'
test_images  = 'test_images'

_columns = ['company_name', 'full_name', 'position_name', 'address', 'phone_number', 'fax', 'mobile', 'email', 'url']

_img_len = 96

_batch_size = 128
_nb_epoch   = 20
_sgd_lr     = 0.1
_sgd_decay  = 0.001
_Wreg_l2    = 0.0001

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)


def train():
    from keras.optimizers import SGD
    from keras.preprocessing.image import ImageDataGenerator

    logging.info('... building model')

    sgd = SGD(lr=_sgd_lr, decay=_sgd_decay, momentum=0.9, nesterov=True)

    model = resnet()
    model.compile(
        loss=_objective,
        optimizer=sgd,
        metrics=['mae'])

    logging.info('... loading data')

    X, Y = load_train_data()

    logging.info('... training')

    datagen = ImageDataGenerator(
        # data augmentation
        width_shift_range  = 1./8.,
        height_shift_range = 1./8.,
        rotation_range     = 0.,
        shear_range        = 0.,
        zoom_range         = 0.,
    )

    model.fit_generator(
        datagen.flow(X, Y, batch_size=_batch_size),
        samples_per_epoch=X.shape[0],
        nb_epoch=_nb_epoch,
        verbose=1)

    return model


def load_train_data():
    df = pd.read_csv(train_csv)
    X = _load_rawdata(df, train_images)
    Y = df[_columns].values
    return X, Y


def resnet(repetition=2, k=1):
    '''Wide Residual Network (with a slight modification)
    depth == repetition*6 + 2
    '''
    from keras.models import Model
    from keras.layers import Input, Dense, Flatten, AveragePooling2D
    from keras.regularizers import l2

    input_shape = (1, _img_len, _img_len)
    output_dim = len(_columns)

    x = Input(shape=input_shape)

    z = conv2d(nb_filter=8, k_size=5, downsample=True)(x)        # out_shape ==    8, _img_len/ 2, _img_len/ 2
    z = bn_lrelu(0.01)(z)
    z = residual_block(nb_filter=k*16, repetition=repetition)(z) # out_shape == k*16, _img_len/ 4, _img_len/ 4
    z = residual_block(nb_filter=k*32, repetition=repetition)(z) # out_shape == k*32, _img_len/ 8, _img_len/ 8
    z = residual_block(nb_filter=k*64, repetition=repetition)(z) # out_shape == k*64, _img_len/16, _img_len/16
    z = AveragePooling2D((_img_len/16, _img_len/16))(z)
    z = Flatten()(z)
    z = Dense(output_dim=output_dim, activation='sigmoid', W_regularizer=l2(_Wreg_l2), init='zero')(z)

    return Model(input=x, output=z)


def residual_block(nb_filter, repetition):
    '''(down dample ->) residual blocks ....... -> BatchNormalization -> LeakyReLU'''
    from keras.layers import merge
    def f(x):
        for i in xrange(repetition):
            if i == 0:
                y = conv2d(nb_filter, downsample=True, k_size=1)(x)
                z = conv2d(nb_filter, downsample=True)(x)
            else:
                y = x
                z = bn_lrelu(0.01)(x)
                z = conv2d(nb_filter)(z)
            z = bn_lrelu(0.01)(z)
            z = conv2d(nb_filter)(z)
            x = merge([y, z], mode='sum')
        return bn_lrelu(0.01)(x)
    return f


def bn_lrelu(alpha):
    '''BatchNormalization -> LeakyReLU'''
    from keras.layers import BatchNormalization, LeakyReLU
    def f(x):
        return LeakyReLU(alpha)(BatchNormalization(mode=0, axis=1)(x))
    return f


def conv2d(nb_filter, k_size=3, downsample=False):
    from keras.layers import Convolution2D
    from keras.regularizers import l2
    def f(x):
        subsample = (2, 2) if downsample else (1, 1)
        border_mode = 'valid' if k_size == 1 else 'same'
        return Convolution2D(
                    nb_filter=nb_filter, nb_row=k_size, nb_col=k_size, subsample=subsample,
                    init='glorot_normal', W_regularizer=l2(_Wreg_l2), border_mode=border_mode)(x)
    return f


def _objective(y_true, y_pred):
    '''最適化したい誤差関数'''
    from keras import backend as K
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)


def _save_model(model, fn):
    with open(fn + '.json', 'w') as f:
        f.write(model.to_json())
    model.save_weights(fn + '.h5', overwrite=True)


def _load_model(fn):
    from keras.models import model_from_json
    with open(fn + '.json') as f:
        model = model_from_json(f.read())
    model.load_weights(fn + '.h5')
    return model


def _load_rawdata(df, dir_images):
    '''画像を読み込み、4Dテンソル (len(df), 1, _img_len, _img_len) として返す
    '''

    X = np.zeros((len(df), 1, _img_len, _img_len), dtype=np.float32)

    for i, row in df.iterrows():
        img = Image.open(os.path.join(dir_images, row.filename))
        img = img.crop((row.left, row.top, row.right, row.bottom))
        img = img.convert('L')
        img = img.resize((_img_len, _img_len), resample=Image.BICUBIC)

        # 白黒反転しつつ最大値1最小値0のfloat32に画素値を正規化
        img = np.asarray(img, dtype=np.float32)
        b, a = np.max(img), np.min(img)
        X[i, 0] = (b-img) / (b-a) if b > a else 0

    return X


def predict(model):
    logging.info('... predicting')
    df = pd.read_csv(test_csv, index_col=0)
    X = _load_rawdata(df, test_images)
    P = model.predict(X, verbose=1)
    pd.DataFrame(P).to_csv(fn_prediction, header=False, float_format='%.6f')


if __name__ == '__main__':
    model = train()
    predict(model)

    ## モデルを画像出力させたい時コメントアウトを外す
    # from keras.utils.visualize_util import plot
    # plot(model, to_file='model.png', show_shapes=True)

    ## モデルをファイル保存させたい時コメントアウトを外す
    # _save_model(model, 'saved_model')
