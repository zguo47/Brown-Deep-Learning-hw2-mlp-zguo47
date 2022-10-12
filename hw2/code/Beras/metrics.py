import numpy as np

from .core import Callable


class CategoricalAccuracy(Callable):
    def forward(self, probs, labels):
        """Categorical accuracy forward pass!"""
        super().__init__()
        # TODO: Compute and return the categorical accuracy of your model given the output probabilities and true labels
        # max = np.array([np.argmax(prb) for prb in probs])
        return np.mean(np.argmax(probs, axis=1) == np.argmax(labels, axis=1))