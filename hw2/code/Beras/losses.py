import numpy as np

from .core import Diffable


class MeanSquaredError(Diffable):
    def __init__(self):
        super().__init__()
        self.y_pred  = None
        self.y_true  = None
        self.outputs = None

    def forward(self, y_pred, y_true):
        """Mean squared error forward pass!"""
        # TODO: Compute and return the MSE given predicted and actual labels
        self.y_pred = np.array(y_pred)
        self.y_true = np.array(y_true)

        # Your code here:
        self.outputs = np.mean((self.y_pred-self.y_true)**2)
        return self.outputs

    def input_gradients(self):
        """Mean squared error backpropagation!"""
        # TODO: Compute and return the gradients
        return 2 * (self.y_pred-self.y_true)


def clip_0_1(x, eps=1e-8):
    return np.clip(x, eps, 1-eps)

class CategoricalCrossentropy(Diffable):
    """
    GIVEN. Feel free to use categorical cross entropy as your loss function for your final model.
    """

    def __init__(self):
        super().__init__()
        self.truths  = None
        self.inputs  = None
        self.outputs = None

    def forward(self, inputs, truths):
        """Categorical cross entropy forward pass!"""
        # print(truth.shape, inputs.shape)
        ll_right = truths * np.log(clip_0_1(inputs))
        ll_wrong = (1 - truths) * np.log(clip_0_1(1 - inputs))
        nll_total = -np.mean(ll_right + ll_wrong, axis=-1)

        self.inputs = inputs
        self.truths = truths
        self.outputs = np.mean(nll_total, axis=0)
        return self.outputs

    def input_gradients(self):
        """Categorical cross entropy backpropagation!"""
        bn, n = self.inputs.shape
        grad = np.zeros(shape=(bn, n), dtype=self.inputs.dtype)
        for b in range(bn):
            inp = self.inputs[b]
            tru = self.truths[b]
            grad[b] = inp - tru
            grad[b] /= clip_0_1(inp - inp ** 2)
            grad[b] /= inp.shape[-1]
        return grad
