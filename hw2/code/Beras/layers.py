from tkinter import W
from turtle import shape
from xml.sax.xmlreader import InputSource
import numpy as np

from .core import Diffable


class Dense(Diffable):
    def __init__(self, input_size, output_size, initializer="kaiming"):
        super().__init__()
        self.w, self.b = self.__class__._initialize_weight(
            initializer, input_size, output_size
        )
        self.weights = [self.w, self.b]
        self.inputs  = None
        self.outputs = None

    def forward(self, inputs):
        """Forward pass for a dense layer! Refer to lecture slides for how this is computed."""
        self.inputs = inputs

        # TODO: implement the forward pass and return the outputs
        self.outputs = np.matmul(inputs, self.weights[0]) + self.weights[1]
        return self.outputs

    def weight_gradients(self):
        """Calculating the gradients wrt weights and biases!"""
        # TODO: Implement calculation of gradients
        wgrads = []
        for i in self.inputs:
            m = np.array([i,] * self.w.shape[-1]).transpose()
            wgrads.append(m)
        wgrads = np.array(wgrads)
        bgrads = np.ones(self.w.shape[1])
        return wgrads, bgrads

    def input_gradients(self):
        """Calculating the gradients wrt inputs!"""
        # TODO: Implement calculation of gradients        
        return self.w

        

    @staticmethod
    def _initialize_weight(initializer, input_size, output_size):
        """
        Initializes the values of the weights and biases. The bias weights should always start at zero.
        However, the weights should follow the given distribution defined by the initializer parameter
        (zero, normal, xavier, or kaiming). You can do this with an if statement
        cycling through each option!

        Details on each weight initialization option:
            - Zero: Weights and biases contain only 0's. Generally a bad idea since the gradient update
            will be the same for each weight so all weights will have the same values.
            - Normal: Weights are initialized according to a normal distribution.
            - Xavier: Goal is to initialize the weights so that the variance of the activations are the
            same across every layer. This helps to prevent exploding or vanishing gradients. Typically
            works better for layers with tanh or sigmoid activation.
            - Kaiming: Similar purpose as Xavier initialization. Typically works better for layers
            with ReLU activation.
        """

        initializer = initializer.lower()
        assert initializer in (
            "zero",
            "normal",
            "xavier",
            "kaiming",
        ), f"Unknown dense weight initialization strategy '{initializer}' requested"
        io_size = (input_size, output_size)

        # TODO: Implement default assumption: zero-init for weights and bias

        # TODO: Implement remaining options (normal, xavier, kaiming initializations). Note that
        # strings must be exactly as written in the assert above
        weights = np.zeros(io_size)
        bias = np.zeros(output_size)

        if initializer == "Normal":
            weights = np.random.normal(0, 1, io_size)
            bias = np.random.normal(0, 1, output_size)
        
        if initializer == "Xavier":
            weights = np.random.normal(0, np.sqrt(2/(input_size + output_size)), io_size)
            bias = np.random.normal(0, np.sqrt(2 / (input_size + output_size)), output_size)
        
        if initializer == "Kaiming":
            weights = np.random.normal(0, np.sqrt(2/input_size), io_size)
            bias = np.random.normal(0, np.sqrt(2/input_size), output_size)

        # bias = bias.reshape(1, output_size)
        return weights, bias
