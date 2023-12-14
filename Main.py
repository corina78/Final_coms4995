import numpy as np
from os.path import join
import pickle
from loader import MnistDataloader
from preprocessing import prepare_data, one_hot_encode
from helper_functions import *


def Model(X, X_test, Y, Y_test, layers_dims, learning_rate=0.01, num_iterations=1):
    """
    Parameters:
    X -- input dataset, shaped (num_features, num_examples)
    Y -- truth "label" array, shaped (1, total examples)
    units_in_layer -- list holding the size of the input and each layer, having length (total layers + 1).
    learning_rate -- rate of learning for the gradient descent updating process
    num_iterations -- total cycles of the optimization procedure
    print_cost -- if set to True, the cost is displayed every 50 intervals

    Outputs:
    parameters -- model's trained parameters. These can subsequently be used for predictions.
    """

    np.random.seed(1)
    costs_train = []  # keep track of cost
    costs_test = []
    iterations = []  # keep track of iterations

    # Parameters initialization.
    print("Initializing parameters...")
    parameters = initialize_parameters(units_in_layer)
    print("Parameters initialized!")

    # Loop (gradient descent)
    for i in range(0, num_iterations):
        print("\n=== Iteration {} ===".format(i))

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SOFTMAX.
        print("Starting forward propagation...")
        AL, caches = Model_forward(X, parameters)
        # Forward propagation for testing examples
        AL_test, caches_test = Model_forward(X_test, parameters)
        print("Forward propagation done!")

        # Compute cost for training examples.
        cost_train = compute_cost(AL, Y)
        # Compute cost for testing examples
        cost_test = compute_cost(AL_test,Y_test)
        print("Cost computed!")

        # Backward propagation.
        print("Starting backward propagation...")
        grads = Model_backward(AL, Y, caches)
        print("Backward propagation done!")

        # Update parameters.
        print("Updating parameters...")
        parameters = update_parameters(parameters, grads, learning_rate)
        print("Parameters updated!")

        costs_train.append(cost_train)
        costs_test.append(cost_test)
        iterations.append(i)



    return parameters, costs_train, costs_test, iterations


def predict(X, parameters):
    """
    Given input features and parameters, it predicts the class labels

    Arguments:
    X -- input features, numpy array of shape (number of features, number of examples)
    parameters -- python dictionary containing the updated parameters of the model

    Returns:
    predictions -- vector of predicted labels for the examples in X
    """

    # Forward propagation
    AL, caches = Model_forward(X, parameters)

    # Convert probabilities AL into a prediction by taking the class with the highest probability
    predictions = np.argmax(AL, axis=0)

    return predictions


def compute_accuracy(predictions, Y):
    """
    Computes the accuracy of the predictions against the true labels

    Arguments:
    predictions -- predicted labels, a numpy array of shape (1, number of examples)
    Y -- true "label" vector (containing labels from 0 to 9), of shape (1, number of examples)

    Returns:
    accuracy -- accuracy of the predictions
    """
    accuracy = np.mean(predictions == Y)
    return accuracy


if __name__=="__main__":

    # Call relevant functions in order

    input_path = '/home/corina/Documents/Math_Machine_Learning/minst'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                       test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Preprocess datasets
    x_train_flattened, x_test_flattened, y_train_flattened, y_test_flattened = prepare_data(x_train, y_train, x_test,
                                                                                            y_test)
    # One hot encode Y ground true values
    one_hot_encoded_y_train = one_hot_encode(y_train_flattened)
    one_hot_encoded_y_test = one_hot_encode(y_test_flattened)

    # Define the number of units in each layer of the network
    units_in_layer = [784,256,128,10]

    parameters, costs_train, costs_test, iterations = Model(x_train_flattened, x_test_flattened, one_hot_encoded_y_train.T, one_hot_encoded_y_test.T, units_in_layer,learning_rate=0.01, num_iterations=100)

    # Data to save
    data_to_save = {
        'costs_train': costs_train,
        'costs_test': costs_test,
        'iterations': iterations
    }

    # Save the data to a file
    with open('model_costsMinst' + str(units_in_layer) + str(0.01) + '.pkl', 'wb') as file:
        pickle.dump(data_to_save, file)

    print("Data saved successfully.")

    predictions_train = predict(x_train_flattened, parameters)
    predictions_test = predict(x_test_flattened, parameters)

    accuracy_train = compute_accuracy(predictions_train, y_train_flattened)
    accuracy_test = compute_accuracy(predictions_test, y_test_flattened)
    print(accuracy_train)
    print(accuracy_test)