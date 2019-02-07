import numpy as np
from keras import Input, Model
from keras.optimizers import Adam
from tqdm.auto import trange
from tqdm import trange

from . import vis


class GAN:
    default_optimizer = Adam(lr=0.0002, beta_1=0.5)

    def __init__(self, generator, discriminator,
                 generator_optimizer=default_optimizer, discriminator_optimizer=default_optimizer,
                 noise='normal', noise_params=None):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.noise_input_shape = generator.layers[0].input_shape[1:]
        self.__combine_discriminator_generator(generator, discriminator)
        self.set_noise(noise, noise_params)

    def set_noise(self, noise, noise_params):
        if noise_params is None and noise == 'normal':
            self.noise_params = {'mu': 0, 'sigma': 1}
        elif noise_params is None and noise == 'uniform':
            self.noise = {'min': 0, 'max': 1}
        self.noise = noise

    def generate(self, nr):
        noise = self.__generate_noise((nr,) + self.noise_input_shape)
        return self.generator.predict(noise)

    def __combine_discriminator_generator(self, generator, discriminator):
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=self.discriminator_optimizer, metrics=['acc'])
        gan_noise_input = Input(shape=self.noise_input_shape, name='gan_noise_input')
        generator_out = generator(inputs=[gan_noise_input])
        discriminator.trainable = False
        gan_output = discriminator(inputs=[generator_out])
        self.gan = Model(inputs=[gan_noise_input], outputs=[gan_output], name='GAN')
        self.gan.compile(loss='binary_crossentropy', optimizer=self.generator_optimizer, metrics=['acc'])

    def __generate_noise(self, shape):
        if callable(self.noise):
            return self.noise(shape)
        if self.noise is 'normal':
            return np.random.normal(self.noise_params['mu'], self.noise_params['sigma'], shape)
        return np.random.uniform(self.noise_params['min'], self.noise_params['max'], shape)

    def train_random_batches(self, X, Y=None, batches=1000, batch_size=32, log_interval=1, plot_interval=50):
        if batch_size >= X.shape[0]:
            batch_size = X.shape[0]

        with trange(batches) as prog_bar:
            for i in prog_bar:

                # Input for the generator
                noise_input_batch = self.__generate_noise((batch_size,) + self.noise_input_shape)
                # Generate fake inputs
                generator_predictions = self.generator.predict([noise_input_batch], batch_size=batch_size)
                # Retrieve real examples
                batch_indexes = np.random.randint(0, X.shape[0], batch_size)
                x_batch = X[batch_indexes]
                d_accuracy, d_loss = self.train_discriminator(generator_predictions, x_batch)

                g_accuracy, g_loss = self.train_generator(batch_size)

                if log_interval != 0 and (i % log_interval == 0):
                    prog_bar.set_description("Batch " + str(i + 1) + ",  " + " D loss: " + str(round(d_loss, 4)) +
                                             " D acc: " + str(round(d_accuracy, 4)) +
                                             " G loss: " + str(round(g_loss, 4)) +
                                             " G acc: " + str(round(g_accuracy, 4)))
                if plot_interval != 0 and (i % plot_interval == 0):
                    vis.show_gan_image_predictions(self, 32)

    def train(self, X, Y=None, epochs=10, batch_size=32, log_interval=1, plot_interval=50):
        if batch_size >= X.shape[0]:
            batch_size = X.shape[0]

        batches_done = 0
        batch_count = X.shape[0] // batch_size
        with trange(epochs) as prog_bar:
            for i in prog_bar:
                D_accs = []
                D_losses = []
                G_accs = []
                G_losses = []
                for j in range(batch_count):
                    # Input for the generator
                    noise_input_batch = self.__generate_noise((batch_size,) + self.noise_input_shape)
                    # Generate fake inputs
                    generator_predictions = self.generator.predict([noise_input_batch], batch_size=batch_size)
                    # Retrieve real examples
                    batch_indexes = range(j * batch_size, (j + 1) * batch_size)
                    x_batch = X[batch_indexes]

                    d_accuracy, d_loss = self.train_discriminator(generator_predictions, x_batch)

                    g_accuracy, g_loss = self.train_generator(batch_size)

                    D_accs.append(d_accuracy)
                    D_losses.append(d_loss)
                    G_accs.append(g_accuracy)
                    G_losses.append(g_loss)
                    batches_done = batches_done + 1
                    if log_interval != 0 and (batches_done % log_interval == 0):
                        prog_bar.set_description("Epoch " + str(i + 1) + ",  " + " D loss: " + str(round(d_loss, 4)) +
                                                 " D acc: " + str(round(d_accuracy, 4)) +
                                                 " G loss: " + str(round(g_loss, 4)) +
                                                 " G acc: " + str(round(g_accuracy, 4)))
                    if plot_interval != 0 and (batches_done % plot_interval == 0):
                        vis.show_gan_image_predictions(self, 32)

    def train_generator(self, batch_size):
        # Train the generator, 2* batch size to get the same nr as the discriminator
        noise_input_batch = self.__generate_noise((2 * batch_size,) + self.noise_input_shape)
        y_generator = [1] * 2 * batch_size
        g_loss, g_accuracy = self.gan.train_on_batch(x=[noise_input_batch], y=y_generator)
        return g_accuracy, g_loss

    def train_discriminator(self, fake_batch, x_batch):
        # labels for the discriminator
        y_real_discriminator = [1] * x_batch.shape[0]
        y_fake_discriminator = [0] * fake_batch.shape[0]
        # Train the discriminator
        (d_loss1, d_accuracy1) = self.discriminator.train_on_batch(x=[x_batch],
                                                                   y=y_real_discriminator)
        (d_loss2, d_accuracy2) = self.discriminator.train_on_batch(x=[fake_batch],
                                                                   y=y_fake_discriminator)
        d_loss = (d_loss1 + d_loss2) / 2
        d_accuracy = (d_accuracy1 + d_accuracy2) / 2
        return d_accuracy, d_loss
