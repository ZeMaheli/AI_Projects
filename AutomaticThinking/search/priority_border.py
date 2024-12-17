from queue import PriorityQueue


class PriorityBorder:
    def __init__(self):
        self.nodes = PriorityQueue()

    def cleanup(self):
        while not self.nodes.empty():
            self.nodes.get()

    def empty(self):
        return self.nodes.empty()

    def insert(self, node_to_insert, priority):
        self.nodes.put(node_to_insert, priority)

    def remove(self):
        return self.nodes.get()
