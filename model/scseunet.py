import  tensorflow as tf
from  tensorflow.keras.layers import  *


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


def sSE_block(inputs):
    x = Conv2D(1, 1, strides=1, padding='same')(inputs)
    #     B,H,W,C----->B,H,W,
    x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.multiply(x, inputs)
    return x


def cSE(inputs, rate=16):
    shape = inputs.shape
    #     print(shape)
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    x = Reshape((1, 1, shape[-1]))(x)
    #     print(x.shape)
    x = Conv2D(shape[-1] // 16, 1, strides=1, padding='same')(x)
    x = Conv2D(shape[-1], 1, strides=1, padding='same')(x)
    x = tf.keras.layers.Activation('sigmoid')(x)
    #     B,1,1,C
    x = tf.multiply(x, inputs)
    return x


def scSE_block(inputs):
    s = sSE_block(inputs)
    c = cSE(inputs, rate=16)
    add = tf.add(s, c)
    return add


def SCSEUNet(num_classes=2, input_shape=(512 * 2, 512 * 2, 3)):
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)

    child_model = back_bone()
    c11, c12, c13, c14, c15 = child_model(input1)
    c21, c22, c23, c24, c25 = child_model(input2)

    conv5 = tf.abs(tf.keras.layers.Subtract()([c25, c15]))
    conv4 = tf.abs(tf.keras.layers.Subtract()([c24, c14]))
    conv3 = tf.abs(tf.keras.layers.Subtract()([c23, c13]))
    conv2 = tf.abs(tf.keras.layers.Subtract()([c22, c12]))
    conv1 = tf.abs(tf.keras.layers.Subtract()([c21, c11]))

    up1 = Conv2DTranspose(512, 3, strides=2, padding='same', activation='relu')(conv5)
    concat1 = tf.concat([up1, conv4], axis=-1)
    conv6 = Conv2D(512, 3, padding='same', activation='relu')(concat1)
    conv6 = Conv2D(512, 3, padding='same', activation='relu')(conv6)
    conv6 = scSE_block(conv6)

    up2 = Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(conv6)
    concat2 = tf.concat([up2, conv3], axis=-1)
    conv7 = Conv2D(256, 3, padding='same', activation='relu')(concat2)
    conv7 = Conv2D(256, 3, padding='same', activation='relu')(conv7)
    conv7 = scSE_block(conv7)

    up3 = Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(conv7)
    concat3 = tf.concat([up3, conv2], axis=-1)
    conv8 = Conv2D(128, 3, padding='same', activation='relu')(concat3)
    conv8 = Conv2D(128, 3, padding='same', activation='relu')(conv8)
    conv8 = scSE_block(conv8)

    up4 = Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(conv8)
    concat4 = tf.concat([up4, conv1], axis=-1)
    conv9 = Conv2D(64, 3, padding='same', activation='relu')(concat4)
    conv9 = Conv2D(64, 3, padding='same', activation='relu')(conv9)
    conv9 = scSE_block(conv9)

    outputs = Conv2D(num_classes, 1, padding='same', activation='softmax')(conv9)
    model = tf.keras.Model(inputs=[input1, input2], outputs=outputs)
    # model.summary()
    return model

