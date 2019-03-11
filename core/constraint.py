from keras.constraints import Constraint
import keras.backend as K


class WeightClipping(Constraint):
    """WeightClipping weight constraint.
    Constraints the weights to be between a lower and an upper bound.
    # Arguments
        c1: lower bound for the weights.
        c2: upper bound for the weights.
    """

    def __init__(self, c1=-0.01, c2=0.01):
        self.c1 = c1
        self.c2 = c2

    def __call__(self, w):
        return K.clip(w, self.c1, self.c2)

    def get_config(self):
        return {'c1': self.c1, 'c2': self.c2}
