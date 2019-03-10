import keras.backend as K


def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    Assumes false samples are -1 and real samples are 1.
    In a standard GAN, the discriminator has a sigmoid output, representing the probability that samples are
    real or generated. In Wasserstein GANs, however, the output should be linear with no
    activation function. The discriminator wants to make the distance between its output
    for real and generated samples as large as possible.
    The most natural way to achieve this is to label generated samples -1 and real
    samples 1, instead of the 0 and 1 used in normal GANs, so that multiplying the
    outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be)
    less than 0."""
    return K.mean(y_true * y_pred)
