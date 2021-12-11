import matplotlib.pyplot as plt
from node import Node


def text_representation_tree(node, interval: str, indent, clear=True):
    if node.name == '':
        return
    if clear:
        source_file = open('tree_text.txt', 'w')
        print('', file=source_file)
        source_file.close()
    view_interval = interval.replace('lambda val: ', '')
    print(indent, '|--', view_interval, '-', node.name, '=', node.result)

    source_file = open('tree_text.txt', 'a')
    print(indent, '|--', view_interval, '-', node.name, '=', node.result, file=source_file)
    source_file.close()

    indent += '    '
    for sub_node in node.sub_nodes:
        text_representation_tree(node.sub_nodes[sub_node], sub_node, indent, False)
