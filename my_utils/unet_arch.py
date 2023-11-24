import tensorflow as tf
from tensorflow.keras import layers


#CNN building blocks

# Convolution block
def conv_block(x, n_filters):
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    return x

#Downsampling
def DS_block(x, n_filters):
    f = conv_block(x, n_filters)
    p = layers.Conv2D(n_filters, 3, 2, padding = "same", activation = "relu", kernel_initializer = "he_normal")(f)
    p = layers.Dropout(0.3)(p)
    return f, p

# Upsampling
def US_block(x, n_filters, conv_features):
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(0.3)(x)
    x = conv_block(x, n_filters)
    return x

#U-Net alike architecture

def build_unet_model(N, M, Ch):
    inputs = layers.Input(shape=(N,M,Ch))

    f1, p1 = DS_block(inputs, 16)
    f2, p2 = DS_block(p1, 32)
    f3, p3 = DS_block(p2, 64)
    f4, p4 = DS_block(p3, 128)
    f5, p5 = DS_block(p4, 256)
    f6, p6 = DS_block(p5, 512)

    bottleneck = conv_block(p6, 1024)

    u6 = US_block(bottleneck, 512, f6)
    u7 = US_block(u6, 256, f5)
    u8 = US_block(u7, 128, f4)
    u9 = US_block(u8, 64, f3)
    u10 = US_block(u9, 32, f2)
    u11 = US_block(u10, 16, f1)
    
    outputs = layers.Conv2D(1, 1, padding="same", activation = "relu")(u11)
    unet_model = tf.keras.Model(inputs, outputs, name="DSE_Net")
    return unet_model
