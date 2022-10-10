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
    subset = subset.lower().strip()
    assert subset in ("test", "train"), f"unknown data subset {subset} requested"
    inputs_file_path, labels_file_path, num_examples = {
        "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 60000),
        "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 10000),
    }[subset]
    inputs_file_path = f"{data_path}/mnist/{inputs_file_path}"
    labels_file_path = f"{data_path}/mnist/{labels_file_path}"

    ## TODO: read the image file and normalize, flatten, and type-convert image
    with open(inputs_file_path, 'rb')as f, gzip.GzipFile(fileobj=f) as bytestream:
        image = np.frombuffer(bytestream.read(-1), np.uint8, -1, 16)
        image = np.array(image).astype(np.float32)/255.0
        image = np.reshape(image, (-1, 784))

    ## TODO: read the label file
    with open(labels_file_path, 'rb')as f, gzip.GzipFile(fileobj=f) as bytestream:
        label = np.frombuffer(bytestream.read(-1), np.uint8, -1, 8)

    return image, label

    
## THE REST ARE OPTIONAL!

'''
def shuffle_data(image_full, label_full, seed):
    pass
    
def get_subset(image_full, label_full, class_list=list(range(10)), num=100):
    pass
'''
