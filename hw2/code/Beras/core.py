from abc import ABC, abstractmethod  # # For abstract method support
from typing import Tuple

import numpy as np


## DO NOT MODIFY THIS CLASS
class Callable(ABC):
    """
    Callable Sub-classes:
     - CategoricalAccuracy (./metrics.py)       - TODO
     - OneHotEncoder       (./preprocess.py)    - TODO
     - Diffable            (.)                  - DONE
    """

    def __call__(self, *args, **kwargs) -> np.array:
        """Lets `self()` and `self.forward()` be the same"""
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> np.array:
        """Pass inputs through function. Can store inputs and outputs as instance variables"""
        pass


## DO NOT MODIFY THIS CLASS
class Diffable(Callable):
    """
    Diffable Sub-classes:
     - Dense            (./layers.py)           - TODO
     - LeakyReLU, ReLU  (./activations.py)      - TODO
     - Softmax          (./activations.py)      - TODO
     - MeanSquaredError (./losses.py)           - TODO
    """

    """Stores whether the operation being used is inside a gradient tape scope"""
    gradient_tape = None  ## All-instance-shared variable

    def __init__(self):
        """Is the layer trainable"""
        super().__init__()
        self.trainable = True  ## self-only instance variable

    def __call__(self, *args, **kwargs) -> np.array:
        """
        If there is a gradient tape scope in effect, perform AND RECORD the operation.
        Otherwise... just perform the operation and don't let the gradient tape know.
        """
        if Diffable.gradient_tape is not None:
            Diffable.gradient_tape.operations += [self]
        return self.forward(*args, **kwargs)

    @abstractmethod
    def input_gradients(self: np.array) -> np.array:
        """Returns gradient for input (this part gets specified for all diffables)"""
        pass

    def weight_gradients(self: np.array) -> Tuple[np.array, np.array]:
        """Returns gradient for weights (this part gets specified for SOME diffables)"""
        return ()

    def compose_to_input(self, J: np.array) -> np.array:
        """
        Compose the inputted cumulative jacobian with the input jacobian for the layer.
        Implemented with batch-level vectorization.

        Requires `input_gradients` to provide either batched or overall jacobian.
        Assumes input/cumulative jacobians are matrix multiplied
        """
        #  print(f"Composing to input in {self.__class__.__name__}")
        ig = self.input_gradients()
        batch_size = J.shape[0]
        n_out, n_in = ig.shape[-2:]
        j_new = np.zeros((batch_size, n_out), dtype=ig.dtype)
        for b in range(batch_size):
            ig_b = ig[b] if len(ig.shape) == 3 else ig
            j_new[b] = ig_b @ J[b]
        return j_new

    def compose_to_weight(self, J: np.array) -> list:
        """
        Compose the inputted cumulative jacobian with the weight jacobian for the layer.
        Implemented with batch-level vectorization.

        Requires `weight_gradients` to provide either batched or overall jacobian.
        Assumes weight/cumulative jacobians are element-wise multiplied (w/ broadcasting)
        and the resulting per-batch statistics are averaged together for avg per-param gradient.
        """
        # print(f'Composing to weight in {self.__class__.__name__}')
        assert hasattr(
            self, "weights"
        ), f"Layer {self.__class__.__name__} cannot compose along weight path"
        J_out = []
        ## For every weight/weight-gradient pair...
        for w, wg in zip(self.weights, self.weight_gradients()):
            batch_size = J.shape[0]
            ## Make a cumulative jacobian which will contribute to the final jacobian
            j_new = np.zeros((batch_size, *w.shape), dtype=wg.dtype)
            ## For every element in the batch (for a single batch-level gradient updates)
            for b in range(batch_size):
                print(b)
                ## If the weight gradient is a batch of transform matrices, get the right entry.
                ## Allows gradient methods to give either batched or non-batched matrices
                wg_b = wg[b] if len(wg.shape) == 3 else wg
                ## Update the batch's Jacobian update contribution
                print("wg_b", wg_b.shape)
                print("J[b]", J[b].shape)

                j_new[b] = wg_b * J[b]
                print("j_new[b]", j_new[b].shape)
            ## The final jacobian for this weight is the average gradient update for the batch
            J_out += [np.mean(j_new, axis=0)]
        ## After new jacobian is computed for each weight set, return the list of gradient updatates
        return J_out


class GradientTape:

    def __init__(self):
        ## Log of operations that were performed inside tape scope
        self.operations = []

    def __enter__(self):
        # When tape scope is entered, let Diffable start recording to self.operation
        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, stop letting Diffable record
        Diffable.gradient_tape = None

    def gradient(self) -> list:
        """Get the gradient from first to last recorded operation"""
        ## TODO:
        ##
        ##  Compute weight gradients for all operations.
        ##  If the model has trainable weights [w1, b1, w2, b2] and ends at a loss L.
        ##  the model should return: [dL/dw1, dL/db1, dL/dw2, dL/db2]
        ##
        ##  Recall that self.operations is populat~ed by Diffable class instances...
        ##
        ##  Start from the last operation and compute jacobian w.r.t input.
        ##  Continue to propagate the cumulative jacobian through the layer inputs
        ##  until all operations have been differentiated through.
        ##
        ##  If an operation that has weights is encountered along the way,
        ##  compute the weight gradients and add them to the return list.
        ##  Remember to check if the layer is trainable before doing this though...

        grads = []
        inputs = self.operations[-1].input_gradients()
        for op in reversed(self.operations):
            if hasattr(op, "weights") and op.trainable:
                grads = (op.compose_to_weight(inputs)) + grads
                
            inputs = op.compose_to_input(inputs)
        return grads
