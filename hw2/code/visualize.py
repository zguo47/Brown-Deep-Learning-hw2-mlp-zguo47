import matplotlib.pyplot as plt
import numpy as np


def visualize_metrics(losses=[], accuracies=[]):
    """
    param losses: a 1D array of loss values
    param accuracies: a 1D array of accuracy values

    Displays a plot with loss and accuracy values on the y-axis and batch number/epoch number on the
    x-axis
    """
    if not losses or not accuracies:
        return print("Must provide a list of losses/accuracies to visualize")
    x = np.arange(1, max(len(losses), len(accuracies)) + 1)
    plt.plot(x, losses)
    plt.plot(x, accuracies)
    plt.ylabel("Loss/Acc Value")
    plt.show()


def visualize_images(model, train_inputs, train_labels_ohe, num_searching=500):
    """
    param model: a neural network model (i.e. SequentialModel)
    param train_inputs: sample training inputs for the model to predict
    param train_labels_ohe: one-hot encoded training labels corresponding to train_inputs

    Displays 10 sample outputs the model correctly classifies and 10 sample outputs the model
    incorrectly classifies
    """

    rand_idx = np.random.choice(len(train_inputs), num_searching)
    rand_batch = train_inputs[rand_idx]
    probs = model.call(rand_batch)

    pred_classes = np.argmax(probs, axis=1)
    true_classes = np.argmax(train_labels_ohe[rand_idx], axis=1)

    right_idx = np.where(pred_classes == true_classes)
    wrong_idx = np.where(pred_classes != true_classes)

    right = np.reshape(rand_batch[right_idx], (-1, 28, 28))
    wrong = np.reshape(rand_batch[wrong_idx], (-1, 28, 28))

    right_pred_labels = true_classes[right_idx]
    wrong_pred_labels = pred_classes[wrong_idx]

    assert len(right) >= 10, f"Found less than 10 correct predictions!"
    assert len(wrong) >= 10, f"Found less than 10 correct predictions!"

    fig, axs = plt.subplots(2, 10)
    fig.suptitle("Classigications\n(PL = Predicted Label)")

    subsets = [right, wrong]
    pred_labs = [right_pred_labels, wrong_pred_labels]

    for r in range(2):
        for c in range(10):
            axs[r, c].imshow(subsets[r][c], cmap="Greys")
            axs[r, c].set(title=f"PL: {pred_labs[r][c]}")
            plt.setp(axs[r, c].get_xticklabels(), visible=False)
            plt.setp(axs[r, c].get_yticklabels(), visible=False)
            axs[r, c].tick_params(axis="both", which="both", length=0)

    plt.show()
