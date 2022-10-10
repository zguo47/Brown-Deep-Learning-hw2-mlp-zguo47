from abc import ABC, abstractmethod
from collections import defaultdict
from pickle import FALSE

import numpy as np

from .core import Diffable


def print_stats(stat_dict, b=None, b_num=None, e=None, avg=False):
    """
    Given a dictionary of names statistics and batch/epoch info,
    print them in an appealing manner. If avg, display stat averages.
    """
    title_str = " - "
    if e is not None:
        title_str += f"Epoch {e+1:2}: "
    if b is not None:
        title_str += f"Batch {b+1:3}"
        if b_num is not None:
            title_str += f"/{b_num}"
    if avg:
        title_str += f"Average Stats"
    print(f"\r{title_str} : ", end="")
    op = np.mean if avg else lambda x: x
    print({k: np.round(op(v), 4) for k, v in stat_dict.items()}, end="")
    print("   ", end="" if not avg else "\n")


def update_metric_dict(super_dict, sub_dict):
    """
    Appends the average of the sub_dict metrics to the super_dict's metric list
    """
    for k, v in sub_dict.items():
        super_dict[k] += [np.mean(v)]


class Model(ABC):

    ###############################################################################################
    ## BEGIN GIVEN

    def __init__(self, layers):
        """
        Initialize all trainable parameters and take layers as inputs
        """
        # Initialize all trainable parameters
        assert all([issubclass(layer.__class__, Diffable) for layer in layers])
        self.layers = layers
        self.trainable_variables = []
        for layer in layers:
            if hasattr(layer, "weights") and layer.trainable:
                for weight in layer.weights:
                    self.trainable_variables += [weight]

    def compile(self, optimizer, loss_fn, acc_fn):
        """
        "Compile" the model by taking in the optimizers, loss, and accuracy functions.
        In more optimized DL implementations, this will have more involved processes
        that make the components extremely efficient but very inflexible.
        """
        self.optimizer = optimizer
        self.compiled_loss = loss_fn
        self.compiled_acc = acc_fn

    def fit(self, x, y, epochs, batch_size):
        """
        Trains the model by iterating over the input dataset and feeding input batches
        into the batch_step method with training. At the end, the metrics are returned.
        """
        agg_metrics = defaultdict(lambda: [])
        batch_num = x.shape[0] // batch_size
        for e in range(epochs):
            epoch_metrics = defaultdict(lambda: [])
            for b, b1 in enumerate(range(batch_size, x.shape[0] + 1, batch_size)):
                b0 = b1 - batch_size
                batch_metrics = self.batch_step(x[b0:b1], y[b0:b1], training=True)
                update_metric_dict(epoch_metrics, batch_metrics)
                print_stats(batch_metrics, b, batch_num, e)
            update_metric_dict(agg_metrics, epoch_metrics)
            print_stats(epoch_metrics, e=e, avg=True)
        return agg_metrics

    def evaluate(self, x, y, batch_size):
        """
        X is the dataset inputs, Y is the dataset labels.
        Evaluates the model by iterating over the input dataset in batches and feeding input batches
        into the batch_step method. At the end, the metrics are returned. Should be called on
        the testing set to evaluate accuracy of the model using the metrics output from the fit method.

        NOTE: This method is almost identical to fit (think about how training and testing differ --
        the core logic should be the same)
        """
        # TODO: Implement evaluate similarly to fit.
        agg_metrics = defaultdict(lambda: [])
        batch_num = x.shape[0] // batch_size
        epoch_metrics = defaultdict(lambda: [])
        for b, b1 in enumerate(range(batch_size, x.shape[0] + 1, batch_size)):
            b0 = b1 - batch_size
            batch_metrics = self.batch_step(x[b0:b1], y[b0:b1], training=FALSE)
            update_metric_dict(epoch_metrics, batch_metrics)
            print_stats(batch_metrics, b, batch_num)
        update_metric_dict(agg_metrics, epoch_metrics)
        print_stats(epoch_metrics, avg=True)
        return agg_metrics

    @abstractmethod
    def call(self, inputs):
        """You will implement this in the SequentialModel class in assignment.py"""
        return

    @abstractmethod
    def batch_step(self, x, y, training=True):
        """You will implement this in the SequentialModel class in assignment.py"""
        return
