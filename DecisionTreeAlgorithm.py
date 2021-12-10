import pandas as pd


class DecisionTree:

    def __init__(self, data, output):
        self.data:pd.DataFrame = data
        self.output = output
        self.attributes = self.data.columns.to_list()
        self.attributes.remove(output)
        print(self.attributes)

    def learn(self):
        print("learn")

    def __learn_tree(self, attributes):
        for attr in attributes:
            print(attr)
