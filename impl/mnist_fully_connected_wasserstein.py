from keras import Sequential, initializers
from keras.datasets import mnist
from keras.layers import LeakyReLU, Dense
from core import GAN, vis, constraint

# Architecture following the fully connected variant described in https://arxiv.org/abs/1701.07875
# Except leaky relu is used instead of relu
mnist_channels = 1
mnist_img_shape = (28, 28, mnist_channels)
mnist_noise_input_shape = (100,)

kernel_initializer = initializers.RandomNormal(stddev=0.02)
weight_clipping = constraint.WeightClipping()

wasserstein_params = {'kernel_initializer': kernel_initializer
    , 'kernel_constraint': weight_clipping, 'bias_constraint': weight_clipping}


def mnist_generator_model():
    generator = Sequential()
    generator.add(Dense(512, input_dim=100, **wasserstein_params))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(512, **wasserstein_params))
    generator.add(LeakyReLU(0.2))
    generator.add(Dense(512, **wasserstein_params))
    generator.add(LeakyReLU(0.2))
    generator.add(
        Dense(784, activation='tanh', **wasserstein_params))
    return generator


def mnist_discriminator_model():
    discriminator = Sequential()
    discriminator.add(Dense(512, input_dim=784, **wasserstein_params))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(512, **wasserstein_params))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(512, **wasserstein_params))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dense(1, **wasserstein_params))
    return discriminator


def mnist_gan():
    return GAN(discriminator=mnist_discriminator_model(), generator=mnist_generator_model(), loss='wasserstein')


if __name__ == '__main__':
    (X_train_mnist, Y_train_mnist), (_, _) = mnist.load_data()
    X_train_mnist = X_train_mnist.reshape((-1, 28 * 28))
    X_train_mnist = X_train_mnist.astype('float32') / 127.5 - 1
    gan = mnist_gan()
    gan.train_random_batches(X_train_mnist, batches=5000, batch_size=32, image_shape=(28, 28), plot_interval=250)
    vis.show_gan_image_predictions(gan, 32, image_shape=(28, 28))
