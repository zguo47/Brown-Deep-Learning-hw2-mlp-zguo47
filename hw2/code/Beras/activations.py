import numpy as np

from .core import Diffable


class LeakyReLU(Diffable):
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        # TODO: Given an input array `x`, compute LeakyReLU(x)
        self.inputs = inputs
        # Your code here:
        new_list = []
        for item in self.inputs:
            ls = np.array([i if i >= 0 else i * self.alpha for i in item])
            new_list.append(ls)
        self.outputs = np.array(new_list)
        return np.array(self.outputs)

    def input_gradients(self):
        # TODO: Compute and return the gradients
        ls = []
        for i, item in enumerate(self.outputs):
            ls.append(item/self.inputs[i])
        return np.array(ls)

    def compose_to_input(self, J):
        # TODO: Maybe you'll want to override the default?
        return super().compose_to_input(J)


class ReLU(LeakyReLU):
    def __init__(self):
        super().__init__(alpha=0)


class Softmax(Diffable):
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        """Softmax forward pass!"""
        # TODO: Implement
        # HINT: Use stable softmax, which subtracts maximum from
        # all entries to prevent overflow/underflow issues
        self.inputs = inputs
        # Your code here:
        self.outputs = None
        return self.outputs

    def input_gradients(self):
        """Softmax backprop!"""
        # TODO: Compute and return the gradients
        return 0
