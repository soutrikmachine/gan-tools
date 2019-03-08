from keras import Sequential, initializers
from keras.datasets import mnist
from keras.layers import Conv2D, LeakyReLU, Dropout, BatchNormalization, Dense, Reshape, UpSampling2D, Activation, \
    Flatten
import numpy as np
from core import GAN, vis

mnist_channels = 1
mnist_img_shape = (28, 28, mnist_channels)
mnist_noise_input_shape = (100,)


def mnist_generator_model():
  generator = Sequential()
  generator.add(Dense(256, input_dim=100, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
  generator.add(LeakyReLU(0.2))
  generator.add(Dense(512))
  generator.add(LeakyReLU(0.2))
  generator.add(Dense(1024))
  generator.add(LeakyReLU(0.2))
  generator.add(Dense(784, activation='tanh'))
  return generator

def mnist_discriminator_model():
  discriminator = Sequential()
  discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
  discriminator.add(LeakyReLU(0.2))
  discriminator.add(Dropout(0.3))
  discriminator.add(Dense(512))
  discriminator.add(LeakyReLU(0.2))
  discriminator.add(Dropout(0.3))
  discriminator.add(Dense(256))
  discriminator.add(LeakyReLU(0.2))
  discriminator.add(Dropout(0.3))
  discriminator.add(Dense(1, activation='sigmoid'))
  return discriminator

def mnist_gan():
    return GAN(discriminator=mnist_discriminator_model(), generator=mnist_generator_model())


if __name__ == '__main__':
    (X_train_mnist, Y_train_mnist), (_, _) = mnist.load_data()
    X_train_mnist = X_train_mnist.reshape((-1, 28 * 28))
    X_train_mnist = X_train_mnist.astype('float32') / 127.5 - 1
    gan = mnist_gan()
    gan.train_random_batches(X_train_mnist, batch_size=32, image_shape=(28,28))
    vis.show_gan_image_predictions(gan, 32)
