import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
def cnn_v2(datanum):
  model = tf.keras.Sequential()

  model.add(layers.Conv1D(128, 3, strides=1, padding='same',input_shape=[datanum, 1]))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha=0.1))
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(256, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha=0.1))
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(512, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha=0.1))
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(1024, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha=0.1))
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(1024, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha=0.1))
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(512, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha=0.1))
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(256, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha=0.1))
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(128, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.LeakyReLU(alpha=0.1))
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(1, 3, strides=1, padding='same', activation='linear'))

  model.build(input_shape=[None, datanum, 1])
  return model
def auto_encoder_v1(datanum):
  num_channels =1
  input_shape=[ datanum, 1]
  input_layer = Input(shape=input_shape)
  conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(input_layer)
  pool1 = MaxPooling1D(pool_size=2)(conv1)
  conv2 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(pool1)
  pool2 = MaxPooling1D(pool_size=2)(conv2)
  conv3 = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(pool2)
  pool3 = MaxPooling1D(pool_size=2)(conv3)

  # Define the decoder layers
  up1 = UpSampling1D(size=2)(pool3)
  conv4 = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(up1)
  up2 = UpSampling1D(size=2)(conv4)
  conv5 = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(up2)
  up3 = UpSampling1D(size=2)(conv5)
  output_layer = Conv1D(filters=1, kernel_size=3, activation='linear', padding='same')(up3)

  # Combine the encoder and decoder layers to form the autoencoder
  autoencoder = Model(inputs=input_layer, outputs=output_layer)

  return autoencoder
def simple_CNN(datanum):
  model = tf.keras.Sequential()

  model.add(layers.Conv1D(64, 3, strides=1, padding='same',input_shape=[ datanum, 1]))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(64, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Conv1D(64, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  #num4
  model.add(layers.Conv1D(64, 3, strides=1, padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.ReLU())
  model.add(layers.Dropout(0.3))

  model.add(layers.Flatten())
  model.add(layers.Dense(datanum))

  model.build(input_shape=[ 1,datanum, 1] )
  #model.summary()

  return model