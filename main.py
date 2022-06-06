# This is a sample Python script.
import pandas as pd
import numpy as np
from DecisionTreeAlgorithm import DecisionTree
from SplitData import train_test_data
from VisualizeTree import text_representation_tree

from BaseFunctions import entropy

import scipy as sc


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from discretize import DiscretizeContinuousAttributes

def run_for(data_base, output):

    # read data from file
    data = pd.read_csv(data_base)

    # split data to test and train samples
    train_data, test_data = train_test_data(data, 70)

    # create a discretizer object
    dca = DiscretizeContinuousAttributes(train_data, output)

    # discretize data for better decision tree
    discrete_data = dca.discretize_data()

    # create a decision tree object
    dta = DecisionTree(discrete_data, output)

    # run the learning algorithm of the decision-tree
    dt = dta.learn()

    # create a simple representation of created decision-tree
    text_representation_tree(dt, '', '')

    # run test for create decision-tree with the test data
    accuracy = dta.test(test_data)

    return accuracy


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    glass_accuracy = run_for('glass.csv', 'Type')
    print('Glass Accuracy:', glass_accuracy)
    heart_accuracy = run_for('heart.csv', 'target')
    print('Heart Accuracy:', heart_accuracy)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

