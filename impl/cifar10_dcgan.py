from keras import Sequential
from keras.datasets import mnist, cifar10
from keras.layers import Conv2D, LeakyReLU, Dropout, BatchNormalization, Dense, Reshape, UpSampling2D, Activation, \
    Flatten
import numpy as np
from core import GAN, vis

cifar10_channels = 3
cifar10_img_shape = (32, 32, cifar10_channels)
cifar10_noise_input_shape = (100,)


def cifar10_generator_model():
    model = Sequential(name='generator')

    model.add(Dense(128 * 8 * 8, activation="relu", input_shape=cifar10_noise_input_shape))
    model.add(Reshape((8, 8, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(cifar10_channels, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))
    return model


def cifar10_discriminator_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=cifar10_img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    #     model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model


def cifar10_dcgan():
    return GAN(discriminator=cifar10_discriminator_model(), generator=cifar10_generator_model())


if __name__ == '__main__':
    (X_train, _), (_, _) = cifar10.load_data()
    X_train = X_train / 127.5 - 1.
    gan = cifar10_dcgan()
    gan.train_random_batches(X_train, batch_size=32)
    vis.show_gan_image_predictions(gan, 32)
