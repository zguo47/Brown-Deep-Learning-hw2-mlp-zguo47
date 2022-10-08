import gzip
import pickle

import numpy as np

"""
TODO: 
Same as HW1. Feel free to copy and paste your old implementation here.
It's a good time to vectorize it, while you're at it!
No need to include CIFAR-specific methods.
"""

def get_data_MNIST(subset, data_path="../data"):
    """
    :param subset: string indicating whether we want the training or testing data 
        (only accepted values are 'train' and 'test')
    :param data_path: directory containing the training and testing inputs and labels
    :return: NumPy array of inputs (float32) and labels (uint8)
    """
    pass
    
## THE REST ARE OPTIONAL!

'''
def shuffle_data(image_full, label_full, seed):
    pass
    
def get_subset(image_full, label_full, class_list=list(range(10)), num=100):
    pass
'''
