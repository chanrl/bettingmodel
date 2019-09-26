"""
Functions and classes to complete non-parametric-learners individual exercise.

Implementation of kNN algorithm modeled on sci-kit learn functionality.

TODO: Improve '__main__' to allow flexible running of script
    (different ks, different number of classes)
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import sys
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
#from src.knn_ploting import plot_distances



def plot_decision_boundary(clf, X, y, n_classes):
    """Plot the decision boundary of a kNN classifier.

    Plots decision boundary for up to 4 classes.

    Colors have been specifically chosen to be color blindness friendly.

    Assumes classifier, clf, has a .predict() method that follows the
    sci-kit learn functionality.

    X must contain only 2 continuous features.

    Function modeled on sci-kit learn example.

    Parameters
    ----------
    clf: instance of classifier object
        A fitted classifier with a .predict() method.
    X: numpy array, shape = [n_samples, n_features]
        Test data.
    y: numpy array, shape = [n_samples,]
        Target labels.
    n_classes: int
        The number of classes in the target labels.
    """
    mesh_step_size = .1

    # Colors are in the order [red, yellow, blue, cyan]
    cmap_light = ListedColormap(['#FFAAAA', '#FFFFAA', '#AAAAFF', '#AAFFFF'])
    cmap_bold = ListedColormap(['#FF0000', '#FFFF00', '#0000FF', '#00CCCC'])

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    feature_1 = X[:, 0]
    feature_2 = X[:, 1]
    x_min, x_max = feature_1.min() - 1, feature_1.max() + 1
    y_min, y_max = feature_2.min() - 1, feature_2.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size),
                         np.arange(y_min, y_max, mesh_step_size))
    dec_boundary = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    dec_boundary = dec_boundary.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, dec_boundary, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(feature_1, feature_2, c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # plt.title(
    #           "{0}-Class classification (k = {1}, metric = '{2}')"
    #           .format(n_classes, clf.k, clf.distance))
    plt.show()


def euclidean_distance(a, b):
    """Compute the euclidean_distance between two numpy arrays.

    Parameters
    ----------
    a: numpy array
    b: numpy array

    Returns
    -------
    numpy array
    """
    return np.sqrt(np.dot(a - b, a - b))


def cosine_distance(a, b):
    """Compute the cosine_distance between two numpy arrays.

    Parameters
    ----------
    a: numpy array
    b: numpy array

    Returns
    -------
    """
    return 1 - np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b))


class KNearestNeighbors(object):
    """Classifier implementing the k-nearest neighbors algorithm.

    Parameters
    ----------
    k: int, optional (default = 5)
        Number of neighbors that get a vote.
    distance: function, optional (default = euclidean)
        The distance function to use when computing distances.
    """

    def __init__(self, k=5, distance=euclidean_distance):
        """Initialize a KNearestNeighbors object."""
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        """Fit the model using X as training data and y as target labels.

        According to kNN algorithm, the training data is simply stored.

        Parameters
        ----------
        X: numpy array, shape = [n_samples, n_features]
            Training data.
        y: numpy array, shape = [n_samples,]
            Target labels.

        Returns
        -------
        None
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Return the predicted labels for the input X test data.

        Assumes shape of X is [n_test_samples, n_features] where n_features
        is the same as the n_features for the input training data.

        Parameters
        ----------
        X: numpy array, shape = [n_samples, n_features]
            Test data.

        Returns
        -------
        result: numpy array, shape = [n_samples,]
            Predicted labels for each test data sample.

        """
        num_train_rows, num_train_cols = self.X_train.shape
        num_X_rows, _ = X.shape
        X = X.reshape((-1, num_train_cols))
        distances = np.zeros((num_X_rows, num_train_rows))
        for i, x in enumerate(X):
            for j, x_train in enumerate(self.X_train):
                distances[i, j] = self.distance(x_train, x)
        # Sort and take top k
        top_k = self.y_train[distances.argsort()[:, :self.k]]
        result = np.zeros(num_X_rows)
        for i, values in enumerate(top_k):
            top_voted_label, _ = Counter(values).most_common(1)[0]
            result[i] = top_voted_label
        return result

    def score(self, X, y_true):
        """Return the mean accuracy on the given data and true labels.

        Parameters
        ----------
        X: numpy array, shape = [n_samples, n_features]
            Test data.
        y_true: numpy array, shape = [n_samples,]
            True labels for given test data, X.

        Returns
        -------
        score: float
            Mean accuracy of self.predict(X) given true labels, y_true.
        """
        y_pred = self.predict(X)
        score = y_true == y_pred
        return np.mean(score)

def plot_mult_decision_boundary(ax, X, y, k, scaled=True, 
                                title='Title', xlabel='xlabel', 
                                ylabel='ylabel', hard_class = True):
    
    """Plot the decision boundary of a kNN classifier.
    
    Builds and fits a sklearn kNN classifier internally.

    X must contain only 2 continuous features.

    Function modeled on sci-kit learn example.

    Parameters
    ----------
    ax: Matplotlib axes object
        The plot to draw the data and boundary on
        
    X: numpy array
        Training data
    
    y: numpy array
        Target labels
    
    k: int
        The number of neighbors that get a vote.
        
    scaled: boolean, optional (default=True)
        If true scales the features, else uses features in original units
    
    title: string, optional (default = 'Title')
        A string for the title of the plot
    
    xlabel: string, optional (default = 'xlabel')
        A string for the label on the x-axis of the plot
    
    ylabel: string, optional (default = 'ylabel')
        A string for the label on the y-axis of the plot
    
    hard_class: boolean, optional (default = True)
        Use hard (deterministic) boundaries vs. soft (probabilistic) boundaries
    

    Returns
    -------
    None
    """
    x_mesh_step_size = 0.1
    y_mesh_step_size = 0.01
    
    #Hard code in colors for classes, one class in red, one in blue
    bg_colors = np.array([np.array([255, 150, 150])/255, np.array([150, 150, 255])/255])
    cmap_light = ListedColormap(bg_colors)
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    
    #Build a kNN classifier
    clf = neighbors.KNeighborsClassifier(n_neighbors=k, weights='uniform')
    
    if scaled:
        #Build pipeline to scale features
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X, y)
    else:
        clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = 45, 85
    y_min, y_max = 2, 4
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, x_mesh_step_size),
                         np.arange(y_min, y_max, y_mesh_step_size))
    if hard_class:
        dec_boundary = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.pcolormesh(xx, yy, dec_boundary, cmap=cmap_light)
        ax.scatter(X[:, 0], X[:, 1], c='black', cmap=cmap_bold)
    else:
        dec_boundary = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        colors = dec_boundary.dot(bg_colors)
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        ax.imshow(colors.reshape(200, 400, 3), origin = "lower", aspect = "auto", extent = (x_min, x_max, y_min, y_max))

    ax.set_title(title + ", k={0}, scaled={1}".format(k, scaled))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))


if __name__ == '__main__':
    X, y = make_classification(n_classes=3, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=1,
                               class_sep=1, random_state=5)
    print(y.shape)

    knn = KNearestNeighbors(4, cosine_distance)
    knn.fit(X, y)
    print("Accuracy: {}".format(knn.score(X, y)))
    print("\tactual\tpredict\tcorrect?")
    for i, (actual, predicted) in enumerate(zip(y, knn.predict(X))):
        print("{}\t{}\t{}\t{}".format(i,
                                  actual,
                                  int(predicted),
                                  int(actual == predicted)))

    # This loop plots the decision boundaries for different decision metrics
    for metric in [euclidean_distance, cosine_distance]:
        # we create an instance of Neighbours Classifier and fit the data.
        clf = KNearestNeighbors(k=3, distance=metric)
        clf.fit(X, y)
        plot_decision_boundary(clf, X, y, n_classes=3)
