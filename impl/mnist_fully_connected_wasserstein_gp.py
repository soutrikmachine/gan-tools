from keras import Sequential
from keras.datasets import mnist
from keras.layers import LeakyReLU, Dense

from core import GAN, vis, WGAN

# Architecture following the fully connected variant described in https://arxiv.org/abs/1701.07875
# Except leaky relu is used instead of relu
mnist_channels = 1
mnist_img_shape = (28, 28, mnist_channels)
mnist_noise_input_shape = (100,)


def mnist_generator_model():
    generator = Sequential()
    generator.add(Dense(512, input_dim=100))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))
    generator.add(
        Dense(784, activation='tanh'))
    return generator


def mnist_discriminator_model():
    discriminator = Sequential()
    discriminator.add(Dense(512, input_dim=784))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(512))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(1))
    return discriminator


def mnist_gan():
    return WGAN(discriminator=mnist_discriminator_model(), generator=mnist_generator_model(), gen_loss='wasserstein', dis_loss = 'wasserstein')


if __name__ == '__main__':
    (X_train_mnist, Y_train_mnist), (_, _) = mnist.load_data()
    X_train_mnist = X_train_mnist.reshape((-1, 28 * 28))
    X_train_mnist = X_train_mnist.astype('float32') / 127.5 - 1
    gan = mnist_gan()
    gan.train_random_batches(X_train_mnist, batches=5000, batch_size=64, image_shape=(28, 28), plot_interval=250
                             ,nr_train_discriminator=5)
    vis.show_gan_image_predictions(gan, 32, image_shape=(28, 28))
