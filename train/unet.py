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

os.makedirs('unet',exist_ok=True)

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

    return a, b


def read_label(mask_path):
    c = tf.io.read_file(mask_path)
    c = tf.image.decode_png(c,channels=3)
    c = tf.image.rgb_to_grayscale(c)
    c = tf.image.resize(c,[512*2,512*2])
    c = tf.cast(c,tf.float32)/255
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


def back_bone():
    inputs = tf.keras.layers.Input(shape=(512 * 2, 512 * 2, 3))

    conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D()(conv1)

    conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D()(conv2)

    conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D()(conv3)

    conv4 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D()(conv4)

    conv5 = tf.keras.layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(conv5)

    #     model=tf.keras.Model(inputs=)
    model = tf.keras.Model(inputs=inputs, outputs=[conv1, conv2, conv3, conv4, conv5])

    return model


def build_model():
    input1 = tf.keras.layers.Input(shape=(512 * 2, 512 * 2, 3))
    input2 = tf.keras.layers.Input(shape=(512 * 2, 512 * 2, 3))

    child_model = back_bone()
    c11, c12, c13, c14, c15 = child_model(input1)
    c21, c22, c23, c24, c25 = child_model(input2)

    conv5 = tf.abs(tf.keras.layers.Subtract()([c25, c15]))
    conv4 = tf.abs(tf.keras.layers.Subtract()([c24, c14]))
    conv3 = tf.abs(tf.keras.layers.Subtract()([c23, c13]))
    conv2 = tf.abs(tf.keras.layers.Subtract()([c22, c12]))
    conv1 = tf.abs(tf.keras.layers.Subtract()([c21, c11]))

    conv5_up = tf.keras.layers.Conv2DTranspose(512,
                                               kernel_size=3,
                                               strides=(2, 2),
                                               padding='same',
                                               activation='relu')(conv5)

    t1_concat = tf.concat([conv4, conv5_up], axis=3)
    conv6 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(t1_concat)
    conv6 = tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu')(conv6)
    conv6_up = tf.keras.layers.Conv2DTranspose(256,
                                               kernel_size=3,
                                               strides=(2, 2),
                                               padding='same',
                                               activation='relu')(conv6)
    t2_concat = tf.concat([conv3, conv6_up], axis=3)

    conv7 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(t2_concat)
    conv7 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(conv7)
    conv7_up = tf.keras.layers.Conv2DTranspose(128,
                                               kernel_size=3,
                                               strides=(2, 2),
                                               padding='same',
                                               activation='relu')(conv7)

    t3_concat = tf.concat([conv2, conv7_up], axis=3)

    conv8 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(t3_concat)
    conv8 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')(conv8)
    conv8_up = tf.keras.layers.Conv2DTranspose(64,
                                               kernel_size=3,
                                               strides=(2, 2),
                                               padding='same',
                                               activation='relu')(conv8)

    t4_concat = tf.concat([conv1, conv8_up], axis=3)

    conv9 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(t4_concat)
    conv9 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(conv9)

    out_put_layer = tf.keras.layers.Conv2D(2, (3, 3), padding='same', activation='softmax')(conv9)

    new_model = tf.keras.models.Model(inputs=[input1, input2],
                                      outputs=out_put_layer)
    new_model.summary()
    # tf.keras.utils.plot_model(new_model)

    return new_model


model=build_model()


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
        name = "unet/epoch_{}_weights.h5".format(self.stopped_epoch)
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
