# This is a sample Python script.
import pandas as pd
from DecisionTreeAlgorithm import DecisionTree

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from discretize import DiscretizeContinuousAttributes

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = pd.read_csv("glass.csv")
    dt = DecisionTree(data, 'Type');
    #discrete_data = DiscretizeContinuousAttributes("glass.csv");
    #discrete_data.discretize_data('Type');

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
