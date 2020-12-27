class Graph:
    def __init__(self, V, E):
        self.V = V
        self.E = E

    @staticmethod
    def from_nodes(V, edge_generator):
        E = []
        for v in V:
            for u in V:
                if u != v:
                    E.append(edge_generator(u, v))

        return Graph(V, E)

class Edge:
    def __init__(self, from_node, to_node, weight):
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight

    def __eq__(self, other):
        return self.from_node == other.from_node and \
            self.to_node == other.to_node

    def __hash__(self):
        return hash((self.from_node, self.to_node))
