from collections import defaultdict
from pprint import pprint

class Graph:
    def __init__(self, V, E):
        for v in V:
            assert isinstance(v, int), v

        for e in E:
            assert isinstance(e, Edge)
            assert e.from_node in V
            assert e.to_node in V

        self.V = V
        self.E = E

        adj = defaultdict(lambda : {})

        for e in E:
            adj[e.from_node][e.to_node] = e.weight

        self.adj_list = adj

        for v in V:
            assert v in adj
            for u in V:
                assert u in adj[v], (type(v), type(u), v, u)

    @staticmethod
    def from_nodes(V, edge_generator):
        E = []

        for v in V:
            for u in V:
                if u != v:
                    E.append(edge_generator(u, v))

        return Graph(V, E)

    def get_edge(self, from_node, to_node):
        return self.adj_list[from_node][to_node].weight

    def __getitem__(self, elem):
        # pprint(elem)
        return self.adj_list[elem]

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
