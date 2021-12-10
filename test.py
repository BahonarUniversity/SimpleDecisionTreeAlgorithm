from node import Node
import pandas as pd

class TestTree:
    def __init__(self, tree, test_data):
        self.tree = tree
        self.test_data = test_data

    def test(self, output):
        sum_of_square_error = 0
        for i in range(self.test_data.shape[0]):
            print("test row: ", self.test_data.iloc[[i]])
            predict = self.tree.predict(self.test_data.iloc[[i]])
            print(type(predict), predict)
            if predict == None:
                continue
            sum_of_square_error += (self.test_data[output] - predict)**2
        return -sum_of_square_error

