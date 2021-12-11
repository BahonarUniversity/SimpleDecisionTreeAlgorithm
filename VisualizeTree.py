import matplotlib.pyplot as plt
from node import Node


def text_representation_tree(node, interval: str, indent):
    if node.name == '':
        return
    view_interval = interval.replace('lambda val: ', '')
    print(indent, '|---', view_interval, '-', node.name, '\n')
    indent += '    '
    for sub_node in node.sub_nodes:
        text_representation_tree(node.sub_nodes[sub_node], sub_node, indent)
