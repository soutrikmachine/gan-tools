import keras
import numpy as np
from keras import Input, Model
from keras.optimizers import Adam
from tqdm.auto import trange
from tqdm import trange
from keras.layers.merge import _Merge
import keras.backend as K
from functools import partial

from util.func import do_n_times
from . import loss as gan_losses
from . import vis


class GAN:

    def __init__(self, generator, discriminator,
                 generator_optimizer=None, discriminator_optimizer=None,
                 noise='normal', noise_params=None, gen_loss='binary_crossentropy', dis_loss = 'binary_crossentropy'):
        if generator_optimizer is None:
            generator_optimizer = Adam(lr=0.0002, beta_1=0.5)
        if discriminator_optimizer is None:
            discriminator_optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.noise_input_shape = generator.layers[0].input_shape[1:]
        self.set_noise(noise, noise_params)
        if dis_loss.lower() == 'wasserstein' or dis_loss is gan_losses.wasserstein_loss:
            dis_loss = gan_losses.wasserstein_loss
            if discriminator.layers[-1].activation != keras.activations.linear and discriminator.layers[
                -1].activation is not None:
                raise ValueError("Wasserstein loss requires the final activation to be linear.")
        if gen_loss.lower() == 'wasserstein' or gen_loss is gan_losses.wasserstein_loss:
            gen_loss = gan_losses.wasserstein_loss
            if discriminator.layers[-1].activation != keras.activations.linear and discriminator.layers[
                -1].activation is not None:
                raise ValueError("Wasserstein loss requires the final activation to be linear.")
        self.dis_loss = dis_loss
        self.gen_loss = gen_loss
        self._combine_discriminator_generator()
        # training statistics
        self.d_losses = []
        self.g_losses = []
        self.d_accs = []
        self.g_accs = []

    def set_noise(self, noise, noise_params):
        if noise_params is None and noise == 'normal':
            self.noise_params = {'mu': 0, 'sigma': 1}
        elif noise_params is None and noise == 'uniform':
            self.noise = {'min': 0, 'max': 1}
        self.noise = noise

    def generate(self, nr):
        noise = self._generate_noise((nr,) + self.noise_input_shape)
        return self.generator.predict(noise)

    def _combine_discriminator_generator(self):
        self.discriminator.compile(loss=self.dis_loss,
                                   optimizer=self.discriminator_optimizer, metrics=['acc'])
        gan_noise_input = Input(shape=self.noise_input_shape, name='gan_noise_input')
        generator_out = self.generator(inputs=[gan_noise_input])
        self.discriminator.trainable = False
        gan_output = self.discriminator(inputs=[generator_out])
        self.gan = Model(inputs=[gan_noise_input], outputs=[gan_output], name='GAN')
        self.gan.compile(loss=self.dis_loss, optimizer=self.generator_optimizer, metrics=['acc'])

    def _generate_noise(self, shape):
        if callable(self.noise):
            return self.noise(shape)
        if self.noise is 'normal':
            return np.random.normal(self.noise_params['mu'], self.noise_params['sigma'], shape)
        return np.random.uniform(self.noise_params['min'], self.noise_params['max'], shape)

    def train_random_batches(self, X, Y=None, batches=1000, batch_size=32, nr_train_discriminator=1,
                             nr_train_generator=1,
                             log_interval=1, plot_interval=50, image_shape=None):
        if batch_size >= X.shape[0]:
            batch_size = X.shape[0]

        with trange(batches) as prog_bar:
            for i in prog_bar:
                # train discriminator nr_train_discriminator times
                d_accuracy, d_loss = do_n_times(nr_train_discriminator, self.train_discriminator_random_batch,
                                                np.mean, X=X, batch_size=batch_size)
                # train generator nr_train_generator times
                g_accuracy, g_loss = do_n_times(nr_train_generator, self.train_generator,
                                                np.mean, batch_size=batch_size)

                if log_interval != 0 and (i % log_interval == 0):
                    prog_bar.set_description("Batch " + str(i + 1) + ",  " + " D loss: " + str(round(d_loss, 4)) +
                                             " D acc: " + str(round(d_accuracy, 4)) +
                                             " G loss: " + str(round(g_loss, 4)) +
                                             " G acc: " + str(round(g_accuracy, 4)))
                    self.d_losses.append(d_loss)
                    self.g_losses.append(g_loss)
                    self.d_accs.append(d_accuracy)
                    self.g_accs.append(g_accuracy)
                if plot_interval != 0 and (i % plot_interval == 0):
                    vis.show_gan_image_predictions(self, 32, image_shape=image_shape)

    def train(self, X, Y=None, epochs=10, batch_size=32, log_interval=1, plot_interval=50, image_shape=None):
        if batch_size >= X.shape[0]:
            batch_size = X.shape[0]

        batches_done = 0
        batch_count = X.shape[0] // batch_size
        with trange(epochs) as prog_bar:
            for i in prog_bar:
                for j in range(batch_count):
                    # Input for the generator
                    noise_input_batch = self._generate_noise((batch_size,) + self.noise_input_shape)
                    # Generate fake inputs
                    generator_predictions = self.generator.predict([noise_input_batch], batch_size=batch_size)
                    # Retrieve real examples
                    batch_indexes = range(j * batch_size, (j + 1) * batch_size)
                    x_batch = X[batch_indexes]

                    d_accuracy, d_loss = self.train_discriminator(generator_predictions, x_batch)

                    g_accuracy, g_loss = self.train_generator(batch_size)

                    batches_done = batches_done + 1
                    if log_interval != 0 and (batches_done % log_interval == 0):
                        prog_bar.set_description("Epoch " + str(i + 1) + ",  " + " D loss: " + str(round(d_loss, 4)) +
                                                 " D acc: " + str(round(d_accuracy, 4)) +
                                                 " G loss: " + str(round(g_loss, 4)) +
                                                 " G acc: " + str(round(g_accuracy, 4)))
                        self.d_losses.append(d_loss)
                        self.g_losses.append(g_loss)
                        self.d_accs.append(d_accuracy)
                        self.g_accs.append(g_accuracy)

                    if plot_interval != 0 and (batches_done % plot_interval == 0):
                        vis.show_gan_image_predictions(self, 32, image_shape=image_shape)

    def train_generator(self, batch_size):
        # Train the generator, 2* batch size to get the same nr as the discriminator
        noise_input_batch = self._generate_noise((2 * batch_size,) + self.noise_input_shape)
        y_generator = [1] * 2 * batch_size
        g_loss, g_accuracy = self.gan.train_on_batch(x=[noise_input_batch], y=y_generator)
        return g_accuracy, g_loss

    def train_discriminator(self, fake_batch, x_batch):
        # labels for the discriminator
        y_real_discriminator = [1] * x_batch.shape[0]
        if self.dis_loss == gan_losses.wasserstein_loss:
            y_fake_discriminator = [-1] * fake_batch.shape[0]
        else:
            y_fake_discriminator = [0] * fake_batch.shape[0]
        # Train the discriminator
        (d_loss1, d_accuracy1) = self.discriminator.train_on_batch(x=[x_batch],
                                                                   y=y_real_discriminator)
        (d_loss2, d_accuracy2) = self.discriminator.train_on_batch(x=[fake_batch],
                                                                   y=y_fake_discriminator)
        d_loss = (d_loss1 + d_loss2) / 2
        d_accuracy = (d_accuracy1 + d_accuracy2) / 2
        return d_accuracy, d_loss

    def train_discriminator_random_batch(self, X, batch_size):
        noise_input_batch = self._generate_noise((batch_size,) + self.noise_input_shape)
        # Generate fake inputs
        generator_predictions = self.generator.predict([noise_input_batch], batch_size=batch_size)
        # Retrieve real examples
        batch_indexes = np.random.randint(0, X.shape[0], batch_size)
        x_batch = X[batch_indexes]
        d_accuracy, d_loss = self.train_discriminator(generator_predictions, x_batch)
        return d_accuracy, d_loss


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        weights = K.random_uniform((K.shape(inputs[0])[0], 1, 1, 1))
        return (weights * inputs[0]) + ((1 - weights) * inputs[1])

class WGAN(GAN):

    def __init__(self, generator, discriminator, generator_optimizer=None,
                 discriminator_optimizer=None, noise='normal', noise_params=None,
                 gen_loss='wasserstein', dis_loss='wasserstein', lam=10.0):
        gen_loss = 'wasserstein'
        dis_loss = 'wasserstein'
        self.lam = lam
        super().__init__(generator, discriminator, generator_optimizer, discriminator_optimizer, noise, noise_params,
                         gen_loss, dis_loss)

    def train_generator(self, batch_size):
        noise_input_batch = self._generate_noise(( batch_size,) + self.noise_input_shape)
        y_generator = [1] * batch_size
        g_loss, g_accuracy = self.gan.train_on_batch(x=[noise_input_batch], y=y_generator)
        return g_accuracy, g_loss

    def train_discriminator(self, fake_batch, x_batch):
        y_real_discriminator = [1.0] * x_batch.shape[0]
        y_fake_discriminator = [-1.0] * fake_batch.shape[0]
        dummy_y = [0.0] * x_batch.shape[0]
        return 0.0, self.discriminator.train_on_batch([x_batch, fake_batch],[y_real_discriminator, y_fake_discriminator, dummy_y])

    def _combine_discriminator_generator(self):
        gan_noise_input = Input(shape=self.noise_input_shape, name='gan_noise_input')
        generator_out = self.generator(inputs=[gan_noise_input])
        self.discriminator.trainable = False
        gan_output = self.discriminator(inputs=[generator_out])
        self.gan = Model(inputs=[gan_noise_input], outputs=[gan_output], name='GAN')
        self.gan.compile(loss=self.dis_loss, optimizer=self.generator_optimizer, metrics=['acc'])
        self.discriminator.trainable = True
        real_samples = Input(shape=self.discriminator.input_shape[1:])
        generated_samples_input = Input(shape=self.discriminator.input_shape[1:])

        discriminator_output_from_generator = self.discriminator(generated_samples_input)
        discriminator_output_from_real_samples = self.discriminator(real_samples)

        averaged_samples = RandomWeightedAverage()([real_samples,generated_samples_input])
        averaged_samples_out = self.discriminator(averaged_samples)

        partial_gp_loss = partial(gan_losses.gp_loss,
                                  averaged_samples=averaged_samples,
                                  gradient_penalty_weight=self.lam)
        partial_gp_loss.__name__ = 'gradient_penalty'

        discriminator_model = Model(inputs=[real_samples,
                                            generated_samples_input],
                                    outputs=[discriminator_output_from_real_samples,
                                             discriminator_output_from_generator,
                                             averaged_samples_out])
        discriminator_model.compile(optimizer=self.discriminator_optimizer,
                                    loss=[gan_losses.wasserstein_loss,
                                          gan_losses.wasserstein_loss,
                                          partial_gp_loss])
        self.discriminator = discriminator_model









