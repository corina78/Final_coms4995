import os
import pickle

def valid_python_identifier(name):
    """ Convert filename into a valid Python identifier by replacing invalid characters. """
    return name.replace('.', '_').replace('-', '_')

def unpickle(file):
    try:
        with open(file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict
    except Exception as e:
        print(f"Error unpickling file {file}: {e}")
        return None

def process_directory(directory):
    data_objects = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        data_dict = unpickle(filepath)
        if data_dict is not None:
            # Create a valid Python identifier from the filename
            identifier = valid_python_identifier(filename)
            data_objects[identifier] = data_dict
    return data_objects

import numpy as np
import matplotlib.pyplot as plt

# CIFAR-10 label names
label_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

def reformat_and_visualize_by_label(data, labels, desired_label, label_names):
    """
    Reformat the flattened image data and visualize the first image found with the desired label.

    :param data: A 10000x3072 numpy array of uint8s.
    :param labels: A list of 10000 numbers in the range 0-9.
    :param desired_label: The label number of the image to visualize.
    :param label_names: A list of label names corresponding to the label numbers.
    """
    # Find the first index with the desired label
    index = labels.index(desired_label)

    # Reshape the data to 32x32x3
    image = data[index].reshape(3, 32, 32).transpose(1, 2, 0)

    # Visualize the image
    plt.imshow(image)
    plt.title(f"Label: {label_names[desired_label]}")
    plt.show()

if __name__ == "__main__":
    # Load the data
    data_objects = process_directory('data/cifar-10-batches-py')
    batch_name = 'data_batch_1'  # Replace with your actual batch name
    batch = data_objects[batch_name]
    data = batch[b'data']  # The data key in your pickle file
    labels = batch[b'labels']  # The labels key in your pickle file
    desired_label = 1  # The label number of the image you want to visualize

    reformat_and_visualize_by_label(data, labels, desired_label, label_names)
