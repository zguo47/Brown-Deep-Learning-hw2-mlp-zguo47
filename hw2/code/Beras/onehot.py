import numpy as np

from .core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        ## TODO: Fetch all the unique labels and create a dictionary with
        ## the unique labels as keys and their one hot encodings as values
        ## HINT: look up np.eye() and see if you can utilize it!

        ## HINT: Wouldn't it be nice if we just gave you the implementation somewhere...

        self.uniq = np.unique(data)  # all the unique labels from `data`
        matrix = np.eye(len(self.uniq))
        self.uniq2oh = {e : matrix[i] for i,e in enumerate(self.uniq)}  # a lookup dictionary with labels and corresponding encodings

    def forward(self, data):
        if not hasattr(self, "uniq2oh"):
            self.fit(data)
        return np.array([self.uniq2oh[x] for x in data])

    def inverse(self, data):
        assert hasattr(self, "uniq"), \
            "forward() or fit() must be called before attempting to invert"
        return np.array([self.uniq[x == 1][0] for x in data])
