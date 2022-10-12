from types import SimpleNamespace

import Beras
from Beras.activations import LeakyReLU
from Beras.core import Diffable
import numpy as np


class SequentialModel(Beras.Model):
    """
    Implemented in Beras/model.py

    def __init__(self, layers):
    def compile(self, optimizer, loss_fn, acc_fn):
    def fit(self, x, y, epochs, batch_size):
    def evaluate(self, x, y, batch_size):           ## <- TODO
    """

    def call(self, inputs):
        """
        Forward pass in sequential model. It's helpful to note that layers are initialized in Beras.Model, and
        you can refer to them with self.layers. You can call a layer by doing var = layer(input).
        """
        # TODO: The call function!
        var = np.copy(inputs)
        for layer in self.layers:
            var = layer(var)
        return var

    def batch_step(self, x, y, training=True):
        """
        Computes loss and accuracy for a batch. This step consists of both a forward and backward pass.
        If training=false, don't apply gradients to update the model! 
        Most of this method (forward, loss, applying gradients)
        will take place within the scope of Beras.GradientTape()
        """
        # TODO: Compute loss and accuracy for a batch.
        # If training, then also update the gradients according to the optimizer
        Diffable.gradient_tape = Beras.GradientTape()
        with Diffable.gradient_tape as tape:
            logits = self.call(x)
            loss = self.compiled_loss.forward(y, logits)
        grads = tape.gradient()
        print(grads[0].shape)
        if training == True:
            self.optimizer.apply_gradients(self.trainable_variables, grads)
        acc = self.compiled_acc.forward(y, logits)
        return {"loss": loss, "acc": acc}


def get_simple_model_components():
    """
    Returns a simple single-layer model.
    """
    ## DO NOT CHANGE IN FINAL SUBMISSION

    from Beras.activations import Softmax
    from Beras.layers import Dense
    from Beras.losses import MeanSquaredError
    from Beras.losses import CategoricalCrossentropy
    from Beras.metrics import CategoricalAccuracy
    from Beras.optimizers import BasicOptimizer
    from Beras.optimizers import RMSProp

    # TODO: create a model and compile it with layers and functions of your choice
    model = SequentialModel([Dense(784, 10),  LeakyReLU(0.1)])
    model.compile(
        optimizer=RMSProp(0.02),
        loss_fn=CategoricalCrossentropy(),
        acc_fn=CategoricalAccuracy(),
    )
    return SimpleNamespace(model=model, epochs=10, batch_size=100)


def get_advanced_model_components():
    """
    Returns a multi-layered model with more involved components.
    """
    # TODO: create/compile a model with layers and functions of your choice.
    from Beras.activations import Softmax
    from Beras.layers import Dense
    from Beras.losses import MeanSquaredError
    from Beras.losses import CategoricalCrossentropy
    from Beras.metrics import CategoricalAccuracy
    from Beras.optimizers import BasicOptimizer
    from Beras.optimizers import RMSProp
    model = SequentialModel([Dense(784, 100),  
    LeakyReLU(0.1), 
    Dense(100, 10),  
    LeakyReLU(0.1)])
    model.compile(
        optimizer=RMSProp(0.02),
        loss_fn=MeanSquaredError(),
        acc_fn=CategoricalAccuracy(),
    )

    return SimpleNamespace(model=model, epochs=10, batch_size=100)


if __name__ == "__main__":
    """
    Read in MNIST data and initialize/train/test your model.
    """
    from Beras.onehot import OneHotEncoder
    import preprocess

    ## Read in MNIST data,
    train_inputs, train_labels = preprocess.get_data_MNIST("train", "../data")
    test_inputs,  test_labels  = preprocess.get_data_MNIST("test",  "../data")

    ## TODO: Use the OneHotEncoder class to one hot encode the labels
    ohe = lambda x: OneHotEncoder().forward(x)  ## placeholder function: returns zero for a given input

    ## Get your model to train and test
    simple = True 
    args = get_simple_model_components() if simple else get_advanced_model_components()
    model = args.model

    ## REMINDER: Threshold of accuracy: 
    ##  1470: >85% on testing accuracy from get_simple_model_components
    ##  2470: >95% on testing accuracy from get_advanced_model_components

    # Fits your model to the training input and the one hot encoded labels
    # This does NOT need to be changed
    train_agg_metrics = model.fit(
        train_inputs, 
        ohe(train_labels), 
        epochs     = args.epochs, 
        batch_size = args.batch_size
    )

    ## Feel free to use the visualize_metrics function to view your accuracy and loss.
    ## The final accuracy returned during evaluation must be > 80%.

    # from visualize import visualize_images, visualize_metrics
    # visualize_metrics(train_agg_metrics["loss"], train_agg_metrics["acc"])
    # visualize_images(model, train_inputs, ohe(train_labels))

    ## Evaluates your model using your testing inputs and one hot encoded labels.
    ## This does NOT need to be changed
    test_agg_metrics = model.evaluate(test_inputs, ohe(test_labels), batch_size=100)
    print('Testing Performance:', test_agg_metrics)
