import tensorflow as tf
import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from tensorflow.keras import backend as K
from tensorflow.keras.layers import  *


EPOCH = 50
BATCH_SIZE = 4

os.makedirs('hrnet',exist_ok=True)

all_images_a_path = glob.glob('D:/LEVIR-CD/train/A/*.png')
all_images_b_path = glob.glob('D:/LEVIR-CD/train/B/*.png')
all_images_mask = glob.glob('D:/LEVIR-CD/train/label/*.png')
'''
设置训练集的路径信息，
第一个路径信息代表时间较早的遥感影像，
第二个时间信息代表时间较新的遥感影像，
第三个路径中的是标签信息。
'''
assert len(all_images_a_path) == len(all_images_b_path) == len(all_images_mask),'训练集三种路径下的有效数据的数量不一致'

test_images_a_path = glob.glob('D:/LEVIR-CD/test/A/*.png')
test_images_b_path = glob.glob('D:/LEVIR-CD/test/B/*.png')
test_images_mask = glob.glob('D:/LEVIR-CD/test/label/*.png')
''''
设置验证集的路径信息，
第一个路径信息代表时间较早的遥感影像，
第二个时间信息代表时间较新的遥感影像，
第三个路径中的是标签信息。
'''
assert len(test_images_a_path) == len(test_images_b_path) == len(test_images_mask),'验证集三种路径下的有效数据的数量不一致'

print('训练集样本数量:{},验证集样本数量:{}。'.format(len(all_images_mask),len(test_images_mask)))


def read_img(a_path, b_path):
    a = tf.io.read_file(a_path)
    a = tf.image.decode_png(a, channels=3)
    a = tf.image.resize(a, [512 * 2, 512 * 2])
    a = tf.cast(a, tf.float32) / 255 - 1

    b = tf.io.read_file(b_path)
    b = tf.image.decode_png(b, channels=3)
    b = tf.image.resize(b, [512 * 2, 512 * 2])
    b = tf.cast(b, tf.float32) / 255 - 1

    concat = tf.concat([a, b], axis=-1)
    return concat


def read_label(mask_path):
    c = tf.io.read_file(mask_path)
    c = tf.image.decode_png(c, channels=3)
    c = tf.image.rgb_to_grayscale(c)
    c = tf.image.resize(c, [512 * 2, 512 * 2])
    c = tf.cast(c, tf.float32) / 255

    return c


dataset_img = tf.data.Dataset.from_tensor_slices((all_images_a_path,all_images_b_path))
dataset_img = dataset_img.map(read_img)
dataset_lab = tf.data.Dataset.from_tensor_slices((all_images_mask))
dataset_lab = dataset_lab.map(read_label)
dataset = tf.data.Dataset.zip((dataset_img,dataset_lab))
dataset = dataset.shuffle(10).batch(BATCH_SIZE).repeat()

test_dataset_img = tf.data.Dataset.from_tensor_slices((test_images_a_path,test_images_b_path))
test_dataset_img = test_dataset_img.map(read_img)
test_dataset_lab = tf.data.Dataset.from_tensor_slices((test_images_mask))
test_dataset_lab = test_dataset_lab.map(read_label)
test_dataset = tf.data.Dataset.zip((test_dataset_img,test_dataset_lab))
test_dataset = test_dataset.batch(BATCH_SIZE)

print('数据格式信息为:{}'.format(dataset))


def conv_bn_relu(inputs, filters, kernel_size=3, strides=1, activate=True):
    conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding='same')(inputs)
    conv = tf.keras.layers.BatchNormalization()(conv)
    if activate:
        conv = tf.keras.layers.Activation('relu')(conv)
    return conv


def conv_block(inputs, filters, strides=1):
    conv = conv_bn_relu(inputs, filters // 4, kernel_size=1, strides=strides)
    conv = conv_bn_relu(conv, filters // 4, kernel_size=3, strides=1)
    conv = conv_bn_relu(conv, filters, kernel_size=1, strides=1, activate=False)

    short = conv_bn_relu(inputs, filters, kernel_size=1, strides=strides, activate=False)

    add = tf.add(conv, short)
    add = tf.keras.layers.Activation('relu')(add)

    return add


def identity_block(inputs, filters):
    conv = conv_bn_relu(inputs, filters // 4, kernel_size=1, strides=1)
    conv = conv_bn_relu(conv, filters // 4, kernel_size=3, strides=1)
    conv = conv_bn_relu(conv, filters, kernel_size=1, strides=1, activate=False)

    add = tf.add(conv, inputs)
    add = tf.keras.layers.Activation('relu')(add)

    return add


def basic_block(inputs, filters):
    conv = conv_bn_relu(inputs, filters, kernel_size=3, strides=1)
    conv = conv_bn_relu(conv, filters, kernel_size=3, strides=1, activate=False)

    add = tf.add(conv, inputs)
    add = tf.keras.layers.Activation('relu')(add)

    return add


def layer1(inputs):
    conv = conv_block(inputs, 256)
    conv = identity_block(conv, 256)
    conv = identity_block(conv, 256)
    conv = identity_block(conv, 256)
    return conv


def transition_layer1(x, out_channels=[32, 64]):
    x0 = conv_bn_relu(x, out_channels[0])
    x1 = conv_bn_relu(x, out_channels[1], strides=2)
    return [x0, x1]


def transition_layer2(x, out_channels=[32, 64, 128]):
    x0 = conv_bn_relu(x[0], out_channels[0])
    x1 = conv_bn_relu(x[1], out_channels[1])
    x2 = conv_bn_relu(x[1], out_channels[2], strides=2)
    return [x0, x1, x2]


def transition_layer3(x, out_channels=[32, 64, 128, 256]):
    x0 = conv_bn_relu(x[0], out_channels[0])
    x1 = conv_bn_relu(x[1], out_channels[1])
    x2 = conv_bn_relu(x[2], out_channels[2])
    x3 = conv_bn_relu(x[2], out_channels[3], strides=2)
    return [x0, x1, x2, x3]


def branch(inputs, channels):
    conv = basic_block(inputs, channels)
    conv = basic_block(conv, channels)
    conv = basic_block(conv, channels)
    conv = basic_block(conv, channels)
    return conv


def fuse_block_1(x):
    '''
    x[0]:down2  32
    x[1]:down4  64
    '''
    x1 = conv_bn_relu(x[1], 32, 1, activate=False)
    x1 = tf.keras.layers.UpSampling2D()(x1)
    x0 = tf.add(x[0], x1)

    x1 = conv_bn_relu(x[0], 64, strides=2, activate=False)
    x1 = tf.add(x1, x[1])

    return [x0, x1]


def fuse_block_2(x):
    '''
    x[0]:down2  32
    x[1]:down4  64
    x[2]:down8  128
    '''
    x11 = x[0]
    x12 = conv_bn_relu(x[1], 32, kernel_size=1, activate=False)
    x12 = tf.keras.layers.UpSampling2D(size=2)(x12)
    x13 = conv_bn_relu(x[2], 32, kernel_size=1, activate=False)
    x13 = tf.keras.layers.UpSampling2D(size=4)(x13)
    x0 = tf.keras.layers.add([x11, x12, x13])

    x21 = conv_bn_relu(x[0], 64, 3, 2, activate=False)
    x22 = x[1]
    x23 = conv_bn_relu(x[2], 64, 1, activate=False)
    x23 = tf.keras.layers.UpSampling2D(size=2)(x23)
    x1 = tf.keras.layers.add([x21, x22, x23])

    x31 = conv_bn_relu(x[0], 32, 3, 2)
    x31 = conv_bn_relu(x31, 128, 3, 2, activate=False)
    x32 = conv_bn_relu(x[1], 128, 3, 2, activate=False)
    x33 = x[2]
    x2 = tf.keras.layers.add([x31, x32, x33])

    return [x0, x1, x2]


def fuse_block_3(x):
    '''
    x[0]:down2  32
    x[1]:down4  64
    x[2]:down8  128
    x[3]:down16 256
    '''
    x0 = x[0]

    x1 = conv_bn_relu(x[1], 32, 1, activate=False)
    x1 = tf.keras.layers.UpSampling2D(size=2)(x1)

    x2 = conv_bn_relu(x[2], 32, 1, activate=False)
    x2 = tf.keras.layers.UpSampling2D(size=4)(x2)

    x3 = conv_bn_relu(x[3], 32, 1, activate=False)
    x3 = tf.keras.layers.UpSampling2D(size=8)(x3)

    out = tf.concat([x0, x1, x2, x3], axis=-1)

    return out


def HRNet(shape=(512 * 2, 512 * 2, 3 * 2), num_classes=2):
    Input = tf.keras.layers.Input(shape=shape)

    conv = conv_bn_relu(Input, 64, strides=2)
    conv = layer1(conv)

    t1 = transition_layer1(conv)

    #     x1
    b10 = branch(t1[0], 32)  # down2
    b11 = branch(t1[1], 64)  # down4
    f1 = fuse_block_1([b10, b11])
    #     x1

    t2 = transition_layer2(f1)

    #     x4
    b20 = branch(t2[0], 32)
    b21 = branch(t2[1], 64)
    b22 = branch(t2[2], 128)
    f2 = fuse_block_2([b20, b21, b22])
    #     x4  仅融合一次，此处的模块可以重复4次

    t3 = transition_layer3(f2)

    #     x3
    b30 = branch(t3[0], 32)
    b31 = branch(t3[1], 64)
    b32 = branch(t3[2], 128)
    b33 = branch(t3[3], 256)
    f3 = fuse_block_3([b30, b31, b32, b33])
    #     x3  仅融合一次，此处的模块可以重复3次

    output = tf.keras.layers.UpSampling2D(size=2)(f3)
    output = conv_bn_relu(output, 64)
    output = tf.keras.layers.Conv2D(num_classes, 1, padding='same', activation='softmax')(output)

    model = tf.keras.Model(inputs=Input, outputs=output)
    model.summary()
    return model


model = HRNet()


def PA(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)

    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.expand_dims(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, tf.int32)

    TP = tf.reduce_sum(tf.cast(y_true * y_pred, tf.int32))
    TN = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.int32))
    FP = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.int32))
    FN = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.int32))

    TP = tf.cast(TP, tf.float32)
    TN = tf.cast(TN, tf.float32)
    FP = tf.cast(FP, tf.float32)
    FN = tf.cast(FN, tf.float32)

    PA = (TP + TN) / (TP + TN + FP + FN + K.epsilon())
    return PA


model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001),
                 loss='sparse_categorical_crossentropy',
                 metrics=['acc',PA])


class MY_EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
    def __init__(self, patience=10):
        super(MY_EarlyStoppingAtMinLoss, self).__init__()
        self.patience = patience * 2
        self.add_time = 0
        self.stopped_epoch = 0
        self.need_stopping = False
        self.all_acc = []

    def on_train_begin(self, logs=None):
        self.best_acc = 0

    def on_epoch_end(self, epoch, logs=None):
        self.stopped_epoch += 1
        current_val_acc = logs.get('val_PA',0)
        self.all_acc.append(current_val_acc)
        name = "hrnet/epoch_{}_weights.h5".format(self.stopped_epoch)
        self.model.save_weights(name)
        if (current_val_acc > self.best_acc) | (current_val_acc == self.best_acc):
            self.best_acc = current_val_acc
            self.add_time = 0
            name = "weights1/epoch_{}_weights.h5".format(self.stopped_epoch)
            self.model.save_weights(name)
            print('\n Model weights1 save to:{}'.format(name), end='')

        else:
            self.add_time += 1
            if (self.add_time > self.patience) | (self.add_time == self.patience):
                self.model.stop_training = True
                self.need_stopping = True

    def on_train_end(self, logs=None):
        if self.need_stopping:
            print("Epoch {}:early stopping".format(self.stopped_epoch))


early_s = MY_EarlyStoppingAtMinLoss(patience=6)

train_sample = len(all_images_mask)
test_sample = len(test_images_mask)

history=model.fit(
                      dataset,
                      epochs=EPOCH,
                      steps_per_epoch=train_sample//BATCH_SIZE,
                      validation_data=test_dataset,
                      validation_steps=test_sample//BATCH_SIZE,
                      callbacks=[early_s]
                   )
# 进行训练，训练50个epochs
