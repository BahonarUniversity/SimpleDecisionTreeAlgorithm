from node import Node
import pandas as pd


class TestTree:
    def __init__(self, tree_algorithm, test_data):
        self.tree_algorithm = tree_algorithm
        self.test_data = test_data

    def test(self, output):
        data_counts = self.test_data.shape[0]
        tp = 0
        for i in range(data_counts):
            row = self.test_data.iloc[[i]]
            predict = self.tree_algorithm.predict(row)
            if predict == '':
                continue
            if predict == row.iloc[0][output]:
                tp += 1
        accuracy = tp / data_counts
        return accuracy

