import tensorflow as tf
import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from tensorflow.keras import backend as K
from tensorflow.keras.layers import  *


EPOCH = 50
BATCH_SIZE = 2
'''
训练的总次数，如果发生显存爆炸，那么则需要减小BATCH_SIZE.00M的错误代表显存爆炸
'''

os.makedirs('deeplab',exist_ok=True)

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


def channel_gate(inputs, rate=16):
    channels = inputs.shape[-1]
    avg_pool = tf.keras.layers.GlobalAveragePooling2D()(inputs)

    fc1 = tf.keras.layers.Dense(channels // rate)(avg_pool)
    fc1 = tf.keras.layers.BatchNormalization()(fc1)
    fc1 = tf.keras.layers.Activation('relu')(fc1)

    fc2 = tf.keras.layers.Dense(channels // rate)(fc1)
    fc2 = tf.keras.layers.BatchNormalization()(fc2)
    fc2 = tf.keras.layers.Activation('relu')(fc2)

    fc3 = tf.keras.layers.Dense(channels)(fc2)

    return fc3


def spatial_gate(inputs, rate=16, d=4):
    channels = inputs.shape[-1]

    conv = tf.keras.layers.Conv2D(channels // rate, 1)(inputs)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation('relu')(conv)

    conv = tf.keras.layers.Conv2D(channels // rate, 3, dilation_rate=d, padding='same')(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation('relu')(conv)

    conv = tf.keras.layers.Conv2D(channels // rate, 3, dilation_rate=d, padding='same')(conv)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation('relu')(conv)

    conv = tf.keras.layers.Conv2D(1, 1)(conv)

    return conv


def BAM_attention(inputs):
    c_out = channel_gate(inputs)
    #     B,C
    s_out = spatial_gate(inputs)

    c_out = tf.keras.layers.RepeatVector(inputs.shape[1] * inputs.shape[2])(c_out)
    #     B,C,H*W
    c_out = tf.reshape(c_out, [-1, inputs.shape[1], inputs.shape[2], inputs.shape[-1]])
    #     B,H,W,C
    out = tf.add(c_out, s_out)
    #     Broadcasting
    out = tf.keras.layers.Activation('sigmoid')(out)
    out = tf.add(tf.multiply(out, inputs), inputs)

    return out


def SKNet_block(inputs, reduce=16):
    conv = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same')(inputs)
    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.ReLU()(conv)

    d1 = tf.keras.layers.Conv2D(256, 1, strides=1, padding='same')(conv)
    d1 = tf.keras.layers.BatchNormalization()(d1)
    d1 = tf.keras.layers.ReLU()(d1)

    d6 = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', dilation_rate=6)(conv)
    d6 = tf.keras.layers.BatchNormalization()(d6)
    d6 = tf.keras.layers.ReLU()(d6)

    d12 = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', dilation_rate=12)(conv)
    d12 = tf.keras.layers.BatchNormalization()(d12)
    d12 = tf.keras.layers.ReLU()(d12)

    d18 = tf.keras.layers.Conv2D(256, 3, strides=1, padding='same', dilation_rate=18)(conv)
    d18 = tf.keras.layers.BatchNormalization()(d18)
    d18 = tf.keras.layers.ReLU()(d18)

    gap = tf.keras.layers.GlobalAvgPool2D()(conv)
    gap = tf.keras.layers.Reshape((1, 1, gap.shape[-1]))(gap)
    gap = tf.keras.layers.Conv2D(256, 1, strides=1, padding='same')(gap)
    gap = tf.keras.layers.BatchNormalization()(gap)
    gap = tf.keras.layers.ReLU()(gap)
    gap = tf.keras.layers.UpSampling2D(size=conv.shape[1])(gap)

    total_features = tf.keras.layers.add([d1, d6, d12, d18, gap])

    total_features = tf.keras.layers.GlobalAvgPool2D()(total_features)

    channels = total_features.shape[-1]

    total_features = tf.keras.layers.Reshape((1, 1, channels))(total_features)

    total_features = tf.keras.layers.Conv2D(int(channels / reduce), 1, strides=1, padding='same')(total_features)
    total_features = tf.keras.layers.BatchNormalization()(total_features)
    total_features = tf.keras.layers.ReLU()(total_features)

    weighs = []

    for i in range(5):
        cur_weight = tf.keras.layers.Conv2D(channels, 1, strides=1, padding='same')(total_features)
        weighs.append(cur_weight)

    concat = tf.keras.layers.concatenate(weighs, axis=-2)
    concat = tf.keras.layers.Softmax(axis=-2)(concat)

    w = []
    for i in range(5):
        cur_w = tf.keras.layers.Cropping2D(cropping=((0, 0), (i, 4 - i)))(concat)
        w.append(cur_w)

    A1 = tf.keras.layers.multiply([d1, w[0]])
    A2 = tf.keras.layers.multiply([d6, w[1]])
    A3 = tf.keras.layers.multiply([d12, w[2]])
    A4 = tf.keras.layers.multiply([d18, w[3]])
    A5 = tf.keras.layers.multiply([gap, w[4]])

    multi_scale_fusion = tf.keras.layers.add([A1, A2, A3, A4, A5])
    multi_scale_fusion = tf.keras.layers.BatchNormalization()(multi_scale_fusion)
    multi_scale_fusion = tf.keras.layers.ReLU()(multi_scale_fusion)

    return multi_scale_fusion


def sSE_block(inputs):
    x = tf.keras.layers.Conv2D(1, 1, strides=1, padding='same')(inputs)
    #     B,H,W,C----->B,H,W,
    x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.multiply(x, inputs)
    return x


def cSE(inputs, rate=16):
    shape = inputs.shape
    #     print(shape)
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = tf.keras.layers.Reshape((1, 1, shape[-1]))(x)
    #     print(x.shape)
    x = tf.keras.layers.Conv2D(shape[-1] // 16, 1, strides=1, padding='same')(x)
    x = tf.keras.layers.Conv2D(shape[-1], 1, strides=1, padding='same')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    #     B,1,1,C
    x = tf.multiply(x, inputs)
    return x


def scSE_block(inputs):
    s = sSE_block(inputs)
    c = cSE(inputs, rate=16)
    add = tf.add(s, c)
    return add


def conv_bn_relu(inputs, filters, kernel_size, strides, activate=True, padding='same', dilate=1):
    conv = tf.keras.layers.Conv2D(filters, kernel_size, strides, padding=padding, dilation_rate=dilate)(inputs)
    conv = tf.keras.layers.BatchNormalization()(conv)
    if activate:
        conv = tf.keras.layers.Activation('relu')(conv)
    return conv


def ASPP(inputs):
    conv = conv_bn_relu(inputs=inputs, filters=256, kernel_size=1, strides=1)

    pool1 = conv_bn_relu(inputs=inputs, filters=256, kernel_size=3, strides=1, dilate=6)
    pool2 = conv_bn_relu(inputs=inputs, filters=256, kernel_size=3, strides=1, dilate=12)
    pool3 = conv_bn_relu(inputs=inputs, filters=256, kernel_size=3, strides=1, dilate=18)

    avg_pool = tf.keras.layers.AveragePooling2D(pool_size=32)(inputs)
    avg_pool = conv_bn_relu(inputs=avg_pool, filters=256, kernel_size=1, strides=1)
    avg_pool = tf.keras.layers.UpSampling2D(size=32)(avg_pool)

    concat = tf.concat([conv, pool1, pool2, pool3, avg_pool], axis=-1)
    return concat


def back_bone(shape=(512 * 2, 512 * 2, 3)):
    Input = tf.keras.layers.Input(shape=shape)

    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same', )(Input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    c = x

    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', )(x)
    residual = BatchNormalization()(residual)
    x = SeparableConv2D(128, (3, 3), padding='same', )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = add([x, residual])

    # x4

    #     print(x.shape)
    #     x=BAM_attention(x)
    c1 = x

    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same')(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(256, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = add([x, residual])

    # x8

    #     x=BAM_attention(x)
    c2 = x
    #     print(x.shape)

    # 如果未使用空洞卷积的话，此处进行降采样处理，否则则使用空洞卷积的dilate代替
    residual = Conv2D(728, (1, 1), strides=2, padding='same')(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu', )(x)
    x = SeparableConv2D(728, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)

    x = add([x, residual])
    c3 = x
    # x16

    for i in range(16):
        residual = x

        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = add([x, residual])
    c4 = x

    # strides=1,not 2
    # so total down 4 times
    #     x=BAM_attention(x)

    residual = Conv2D(1024, (1, 1), strides=1, padding='same')(x)
    residual = BatchNormalization()(residual)
    x = Activation('relu')(x)
    x = SeparableConv2D(728, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, (3, 3), strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    x = add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1536, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(2048, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    c5 = x

    #  input 唯一的一次卷积进行降采样处理
    #  流入 3个block  3次降采样
    #  中间 16个block
    #  输出 2个block  第一个使用残差+降采样，第二个未使用残差

    sk_conv1 = SKNet_block(c5)
    c5 = ASPP(c5)
    conv1 = conv_bn_relu(inputs=c5, filters=256, kernel_size=1, strides=1)

    conv1 = tf.keras.layers.concatenate([conv1, sk_conv1])
    conv1 = conv_bn_relu(inputs=conv1, filters=256, kernel_size=3, strides=1)
    conv1 = conv_bn_relu(inputs=conv1, filters=256, kernel_size=3, strides=1)
    conv1 = scSE_block(conv1)

    model = tf.keras.Model(inputs=Input, outputs=[c, c1, c2, conv1])

    return model


def deeplab_v3_plus(shape=(512 * 2, 512 * 2, 3), num_classes=2):
    input1 = Input(shape=shape)
    input2 = Input(shape=shape)

    child_model = back_bone(shape)

    c11, c12, c13, c14 = child_model(input1)
    c21, c22, c23, c24 = child_model(input2)

    c = tf.abs(tf.keras.layers.Subtract()([c11, c21]))
    c1 = tf.abs(tf.keras.layers.Subtract()([c12, c22]))
    c2 = tf.abs(tf.keras.layers.Subtract()([c13, c23]))
    conv1 = tf.abs(tf.keras.layers.Subtract()([c24, c14]))

    up1 = tf.keras.layers.UpSampling2D(size=2)(conv1)

    concat1 = tf.concat([up1, c2], axis=-1)
    concat1 = conv_bn_relu(inputs=concat1, filters=256, kernel_size=3, strides=1)
    concat1 = conv_bn_relu(inputs=concat1, filters=256, kernel_size=3, strides=1)
    concat1 = scSE_block(concat1)
    #     64*64*256
    up2 = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same')(concat1)

    concat2 = tf.concat([up2, c1], axis=-1)
    concat2 = conv_bn_relu(inputs=concat2, filters=128, kernel_size=3, strides=1)
    concat2 = conv_bn_relu(inputs=concat2, filters=128, kernel_size=3, strides=1)
    concat2 = scSE_block(concat2)

    up3 = tf.keras.layers.Conv2DTranspose(64, 3, 2, padding='same')(concat2)
    concat3 = tf.concat([c, up3], axis=-1)
    concat3 = conv_bn_relu(inputs=concat3, filters=64, kernel_size=3, strides=1)
    concat3 = conv_bn_relu(inputs=concat3, filters=64, kernel_size=3, strides=1)
    concat3 = scSE_block(concat3)

    output = tf.keras.layers.UpSampling2D(size=2)(concat3)
    output = conv_bn_relu(inputs=output, filters=32, kernel_size=3, strides=1)
    output = conv_bn_relu(inputs=output, filters=32, kernel_size=3, strides=1)

    up = Conv2D(num_classes, 1, 1, activation='softmax')(output)

    model = tf.keras.Model(inputs=[input1, input2], outputs=up)
    model.summary()

    return model


model = deeplab_v3_plus()


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
        name = "deeplab/epoch_{}_weights.h5".format(self.stopped_epoch)
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
