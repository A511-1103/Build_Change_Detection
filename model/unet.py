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


def UNET():
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
    # new_model.summary()
    # tf.keras.utils.plot_model(new_model)

    return new_model

