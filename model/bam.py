import  tensorflow as tf
from  tensorflow.keras.layers import  *


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
    x  =BAM_attention(x)
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

    x = BAM_attention(x)
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
    x = BAM_attention(x)

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


def BAM(shape=(512 * 2, 512 * 2, 3), num_classes=2):
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
    # model.summary()

    return model


