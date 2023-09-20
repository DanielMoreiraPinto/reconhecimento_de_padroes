import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D, Add, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Multiply, Activation
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def simple_autoencoder():
    input = Input(shape=(256, 256, 3))

    # Encoder
    x = Conv2D(32, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(input)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(64, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = Conv2DTranspose(128, (3, 3), strides=2, activation="relu", kernel_initializer='he_normal',padding="same")(x)
    x = Conv2DTranspose(64, (3, 3), strides=2, activation="relu",kernel_initializer='he_normal', padding="same")(x)
    x = Conv2DTranspose(32, (3, 3), strides=2, activation="relu", kernel_initializer='he_normal',padding="same")(x)
    x = Conv2D(3, (3, 3), activation="sigmoid", kernel_initializer='he_normal',padding="same")(x)

    # Autoencoder
    autoencoder = Model(input, x)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-03),
                        loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanSquaredError()])
    return autoencoder

def cbd_net():
    input = Input(shape=(256, 256, 3))

    #Noise estimation subnetwork
    x = Conv2D(32, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(input)
    x = Conv2D(32, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(x)
    x = Conv2D(32, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(x)
    x = Conv2D(3, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(x)

    #Non Blind denoising subnetwork
    x = concatenate([x,input])
    conv1 = Conv2D(64, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(x)
    conv2 = Conv2D(64, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv1)

    pool1 = AveragePooling2D(pool_size=(2,2),padding='same')(conv2)
    conv3 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(pool1)
    conv4 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv3)
    conv5 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv4)

    pool2 = AveragePooling2D(pool_size=(2,2),padding='same')(conv5)
    conv6 = Conv2D(256, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(pool2)
    conv7 = Conv2D(256, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv6)
    conv8 = Conv2D(256, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv7)
    conv9 = Conv2D(256, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv8)
    conv10 = Conv2D(256, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv9)
    conv11 = Conv2D(256, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv10)

    upsample1 = Conv2DTranspose(128, (3, 3), strides=2, activation="relu", kernel_initializer='he_normal',padding="same")(conv11)
    add1 = Add()([upsample1,conv5])
    conv12 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(add1)
    conv13 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv12)
    conv14 = Conv2D(128, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv13)

    upsample2 = Conv2DTranspose(64, (3, 3), strides=2, activation="relu", kernel_initializer='he_normal',padding="same")(conv14)
    add1 = Add()([upsample2,conv2])
    conv15 = Conv2D(64, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(add1)
    conv16 = Conv2D(64, (3, 3), activation="relu", kernel_initializer='he_normal',padding="same")(conv15)

    out = Conv2D(3, (1,1), kernel_initializer='he_normal',padding="same")(conv16)
    out = Add()([out,input])

    CBDNet = Model(input,out)
    CBDNet.compile(optimizer=tf.keras.optimizers.Adam(1e-03), loss=tf.keras.losses.MeanSquaredError(),
                   metrics=[tf.keras.metrics.MeanSquaredError()])
    return CBDNet

class EAM(tf.keras.layers.Layer):
  def __init__(self,**kwargs):
    super().__init__(**kwargs)
    
    self.conv1 = Conv2D(64, (3,3), dilation_rate=1,padding='same',activation='relu')
    self.conv2 = Conv2D(64, (3,3), dilation_rate=2,padding='same',activation='relu') 

    self.conv3 = Conv2D(64, (3,3), dilation_rate=3,padding='same',activation='relu')
    self.conv4 = Conv2D(64, (3,3), dilation_rate=4,padding='same',activation='relu')

    self.conv5 = Conv2D(64, (3,3),padding='same',activation='relu')

    self.conv6 = Conv2D(64, (3,3),padding='same',activation='relu')
    self.conv7 = Conv2D(64, (3,3),padding='same')

    self.conv8 = Conv2D(64, (3,3),padding='same',activation='relu')
    self.conv9 = Conv2D(64, (3,3),padding='same',activation='relu')
    self.conv10 = Conv2D(64, (1,1),padding='same')

    self.gap = GlobalAveragePooling2D()

    self.conv11 = Conv2D(64, (3,3),padding='same',activation='relu')
    self.conv12 = Conv2D(64, (3,3),padding='same',activation='sigmoid')

  def call(self,input):
    conv1 = self.conv1(input)
    conv1 = self.conv2(conv1)

    conv2 = self.conv3(input)
    conv2 = self.conv4(conv2)

    concat = concatenate([conv1,conv2])
    conv3 = self.conv5(concat)
    add1 = Add()([input,conv3])

    conv4 = self.conv6(add1)
    conv4 = self.conv7(conv4)
    add2 = Add()([conv4,add1])
    add2 = Activation('relu')(add2)

    conv5 = self.conv8(add2)
    conv5 = self.conv9(conv5)
    conv5 = self.conv10(conv5)
    add3 = Add()([add2,conv5])
    add3 = Activation('relu')(add3)

    gap = self.gap(add3)
    gap = Reshape((1,1,64))(gap)
    conv6 = self.conv11(gap)
    conv6 = self.conv12(conv6)
    
    mul = Multiply()([conv6, add3])
    out = Add()([input,mul]) # This is not included in the reference code
    return out

def rid_net():
    input = Input(shape=(256, 256, 3))

    conv1 = Conv2D(64, (3,3),padding='same')(input)
    eam1 = EAM()(conv1)
    eam2 = EAM()(eam1)
    eam3 = EAM()(eam2)
    eam4 = EAM()(eam3) 
    conv2 = Conv2D(3, (3,3),padding='same')(eam4)
    out = Add()([conv2,input])

    RIDNet = Model(input,out)
    RIDNet.compile(optimizer=tf.keras.optimizers.Adam(1e-03), loss=tf.keras.losses.MeanSquaredError(),
                   metrics=[tf.keras.metrics.MeanSquaredError()])
    return RIDNet
