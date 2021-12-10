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
        #print(self.attributes)

    def learn(self):
        #print("learn")
        self.__learn_tree(self.attributes, self.data, self.tree)
        return self.tree;

    def __learn_tree(self, attributes, dataset, tree_node):

        if type(attributes) == 'NoneType' or len(attributes) == 1:
            tree_node.name = attributes[0]
            tree_node.result = self.get_max_class(dataset, self.output)
            return

        gains = dict();
        for attr in attributes:
            gains[attr] = gain(dataset, attr, self.output)

        max_attr = self.__max_gain_attribute(gains)
        tree_node.set_name(max_attr)
        value_set = set(dataset[max_attr])

        remaining_attributes = attributes.copy()
        remaining_attributes.remove(max_attr)
        #if tree_node.parent_node and tree_node.parent_node.name == "Mg":
        #    print("value_set: ", len(value_set), "  attributes:", remaining_attributes)

        if dataset.shape[0] < 15:
            tree_node.result = self.get_max_class(dataset, self.output)
            return

        if len(value_set) == 1:
            tree_node.result = self.get_max_class(dataset, self.output)
            return
        class_set = set(dataset[self.output]);
        if len(class_set) == 1:
            tree_node.result = class_set.pop()
            return
        #print(value_set)
        for value in value_set:
            #if max_attr == "Ca":
                #print("max_attr: ", max_attr)
            new_node = Node("", tree_node)
            tree_node.add_node(value, new_node)
            self.__learn_tree(remaining_attributes, self.data.loc[self.data[max_attr] == value], new_node)
        #print(value_set)

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
        #print(type(node))
        while type(node) == 'Node':
            node = node.get_valid_node(value)
            if node.result != None and node.result != '':
                return node.result

    def get_max_class(self, dataset, output):
        value_counts = dataset[output].value_counts()
        return value_counts.index[0]
