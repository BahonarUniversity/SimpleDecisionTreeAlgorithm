# This is a sample Python script.
import pandas as pd
from DecisionTreeAlgorithm import DecisionTree
from SplitData import train_test_data
from test import TestTree


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from discretize import DiscretizeContinuousAttributes

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = pd.read_csv("glass.csv")
    train_data, test_data = train_test_data(data, 70)
    dca = DiscretizeContinuousAttributes(train_data, 'Type')
    discrete_data = dca.discretize_data('Type')
    dt = DecisionTree(discrete_data, 'Type')
    dt.learn()

    testT = TestTree(dt, test_data)
    error = testT.test('Type')
    print(error)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
