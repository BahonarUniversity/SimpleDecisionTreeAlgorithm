

class Node:
    def __init__(self, node_name, parent_node, is_it_root=False):
        self.root = is_it_root
        self.name = node_name
        self.sub_nodes = dict()
        self.parent_node = parent_node
        self.result = ''

    def add_node(self, on_value, new_node):
        self.sub_nodes[on_value] = new_node

    def get_node(self, node_name):
        return self.sub_nodes[node_name]

    def set_name(self, name):
        self.name = name

    def get_valid_node(self, value):
        running_value = value.iloc[0][self.name]
        for node in self.sub_nodes:
            check_result = eval(node)(running_value)
            if check_result:
                new_node = self.sub_nodes[node]
                return new_node

        return None

    def sort(self):
        for n in self.sub_nodes:
            self.sub_nodes[n].sort()
        self.sub_nodes = dict(sorted(self.sub_nodes.items()))
