#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a

        # Get prediction of y
        y_hat_i = self.predict(x_i)

        # If prediction is wrong, update weight
        if (y_hat_i != y_i):
            self.W[y_i, :] +=  x_i # Increase weight of gold label
            self.W[y_hat_i, :] -= x_i # Decrease weight of incorrect class


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b
        
        scores = self.W.dot(x_i)[:, None] # Calculate the scores

        softmax = np.exp(scores) / np.sum(np.exp(scores)) # Softmax function
        
        y_one_hot = np.zeros((np.size(self.W, 0), 1)) # One hot encoding [nlabels,1]
        y_one_hot[y_i] = 1 #All the positions of the vector are null except for the position of y_i

        # SGD update. W is num_labels x num_features.
        #self.W -= learning_rate * (softmax - y_one_hot) * x_i[None, : ]
        x_i = x_i.reshape(x_i.shape[0] , 1)
        self.W -= learning_rate * (softmax - y_one_hot).dot(x_i.T)


class MLP(object):
    # Q1.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, n_layers):
        # Initialize an MLP with a single hidden layer. Biases to zero, weights to gaussian
        # with mean 0 and std 0.1.
        self.W1 = np.random.normal(0, 0.1, (hidden_size, n_features))
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.normal(0, 0.1, (n_classes, hidden_size))
        self.b2 = np.zeros(n_classes)

        # Agreggate all weights and biases into lists.
        self.W = [self.W1, self.W2]
        self.b = [self.b1, self.b2]

        self.activation = [self.relu, self.softmax]

        # Store the number of layers.
        self.n_layers = n_layers + 2 # +2 for input and output layers

    def relu(self, input):
        return np.maximum(0, input)

    def softmax(self, input):
        return np.exp(input) / np.sum(np.exp(input))

    def forward_pass(self, x_i):
        # Compute the forward pass. Return values of hidden nodes and output.
        # x_i (n_features): a single training example

        # Initialize the list of hidden nodes.
        hidden_nodes = [x_i]

        # Loop over the layers.
        for i in range(self.n_layers - 1):
            z = np.dot(self.W[i], hidden_nodes[-1]) + self.b[i]
            a = self.activation[i](z)
            
            hidden_nodes.append(a)
        print("hidden_nodes", hidden_nodes[-1])
        return hidden_nodes

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        predictions = [] # y_hat (n_examples)

        # Loop example by example
        for x_i in X:
            # Compute the forward pass.
            hidden_nodes = self.forward_pass(x_i)

            print("hidden_nodes", hidden_nodes[-1])
            # Get the prediction.
            predictions.append(np.argmax(hidden_nodes[-1]))
            print("predictions", predictions[-1])
            exit()

            # Forward pass
            # Hidden layer
            #z1 = self.W1.dot(x_i) + self.b1 # Dot product of weights and inputs plus bias
            #h1 = self.relu(z1) # ReLU activation for hidden layer

            # Output layer
            #z2 = self.W2.dot(h1) + self.b2 # Dot product of weights and inputs plus bias
            #h2 = self.softmax(z2) # Softmax activation for output layer

            # Get prediction
            #predictions.append(np.argmax(h2))

        return predictions

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        print(y_hat[0:5], y[0:5])
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def cross_entropy_loss(self, y, y_pred):
        # Compute the cross-entropy loss. This is the loss function used for
        # multiclass classification.
        """
        y (n_examples): gold labels
        y_pred (n_examples x n_classes): predicted outputs
        """

        # Compute predicted probabilities.
        pred_probabilities = self.softmax(y_pred)

        # Compute the cross-entropy loss.
        loss = -np.sum(y * np.log(pred_probabilities))

        return loss / float(y_pred.shape[0])

    def grad_descent(self, x_i, y_i, learning_rate=0.001):
        # Compute stochastic gradient descent backpropagation. Loss function is
        # cross-entropy. Use the chain rule to compute the gradients of the loss
        # with respect to the weights and biases.
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): rate at which to update weights
        """

        # Forward pass
        hidden_nodes = self.forward_pass(x_i)

        # Cross-entropy loss
        ce_loss = self.cross_entropy_loss(y_i, hidden_nodes[-1])

        # Backward pass
        # Output layer
        dW2 = np.dot(hidden_nodes[-2].T, ce_loss)
        db2 = np.sum(ce_loss, axis=0)

        # Hidden layer
        dW1 = np.dot(hidden_nodes[-3].T, ce_loss)
        db1 = np.sum(ce_loss, axis=0)

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        print("W1", self.W1)
        print("b1", self.b1)
        print("W2", self.W2)
        print("b2", self.b2)
        exit()
        
    def train_epoch(self, X, y, learning_rate=0.001):
        # Train the model for one epoch.
        # X (n_examples x n_features)
        # y (n_examples): gold labels
        # learning_rate (float): rate at which to update weights

        # Loop example by example
        for x_i, y_i in zip(X, y):
            self.grad_descent(x_i, y_i, learning_rate)

def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
        print("Perceptron")
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
