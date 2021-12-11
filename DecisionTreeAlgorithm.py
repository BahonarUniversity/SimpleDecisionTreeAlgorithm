import pandas as pd
from BaseFunctions import gain
from node import Node


class DecisionTree:

    def __init__(self, data, output):
        self.data:pd.DataFrame = data
        self.output = output
        self.attributes = self.data.columns.to_list()
        self.attributes.remove(output)
        self.tree = Node("", None)


    def learn(self) -> object:
        print("Began learning ...")
        self.__learn_tree(self.attributes, self.data, self.tree)
        self.tree.sort()
        return self.tree;

    def __learn_tree(self, attributes, dataset, tree_node):

        gains = dict()
        for attr in attributes:
            gains[attr] = gain(dataset, attr, self.output)

        max_attr = self.__max_gain_attribute(gains)
        tree_node.set_name(max_attr)

        remaining_attributes = attributes.copy()
        remaining_attributes.remove(max_attr)

        if remaining_attributes is None or len(remaining_attributes) == 0 or dataset.shape[0] < 2:
            tree_node.result = self.__get_max_class(dataset, self.output)
            return

        class_set = set(dataset[self.output]);
        if len(class_set) == 1:
            tree_node.result = class_set.pop()
            return

        value_set = set(dataset[max_attr])
        if len(value_set) == 1:
            mystr = ''
            for attr in gains:
                mystr += attr + ': ' + str(gains[attr])+'\n'
            source_file = open('debug_text.txt', 'w')
            print('', file=source_file)
            source_file.close()
            source_file = open('debug_text.txt', 'a')
            print(mystr, file=source_file)
            print(dataset.to_string(), file=source_file)
            tree_node.result = self.__get_max_class(dataset, self.output)
            source_file.close()
            return

        for value in value_set:
            new_node = Node("", tree_node)
            tree_node.add_node(value, new_node)
            self.__learn_tree(remaining_attributes, self.data.loc[self.data[max_attr] == value], new_node)

    def __max_gain_attribute(self, gains):
        max_gain = -1
        target_attr = ""
        for attr in gains:
            if max_gain < gains[attr]:
                max_gain = gains[attr]
                target_attr = attr
        return target_attr

    def predict(self, value):
        node = self.tree.get_valid_node(value)
        result = ''
        if node is None:
            print('node is none:', self.tree.name, 'value: ', value)
            return

        while node is not None:
            if node.result != '':
                result = node.result
                return result
            node = node.get_valid_node(value)
        return result

    def __get_max_class(self, dataset, output):
        value_counts = dataset[output].value_counts()
        return value_counts.index[0]

    def test(self, test_data):
        data_counts = test_data.shape[0]
        tp = 0
        for i in range(data_counts):
            row = test_data.iloc[[i]]
            predict = self.predict(row)
            if predict == '':
                continue
            if predict == row.iloc[0][self.output]:
                tp += 1
        accuracy = tp / data_counts
        return accuracy
