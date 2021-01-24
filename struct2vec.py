import torch

class Struct2vec:
    def __init__(self, mtsp_instance):
        self.mtsp_instance = mtsp_instance
        self.config = mtsp_instance.config
        N = self.config.N

        self.W1 = torch.rand(1, N)
        self.W2 =

    def state(self, input_func):
        G = []
        N = self.config.N
        mtsp_instance = self.mtsp_instance

        graph = mtsp_instance.graph
        depot = mtsp_instance.depot

        W1 = torch.rand(1, N)
        W2 = torch.rand(N, 3)

        for i, v in enumerate(graph.V):
            G.append([])
            for j, u in enumerate(graph.V):
                # append g_ij
                vu_dist = graph[v][u]
                xv_dist = graph[depot][v]
                xu_dist = graph[depot][u]

                u = torch.FloatTensor([vu_dist, xv_dist, xu_dist])
                G[-1].append(torch.matmul(W1, F.relu(torch.matmul(W2, u))))

        T = config.T
        mu_size = config.mu_size

        # random mu initialization

        mu = {}
        next_mu = {}

        for v in mtsp_instance.remaining_cities:
            mu[v] = torch.rand(mu_size, 1)
            next_mu[v] = None

        # main loop for struct2vec
        for t in range(T):
            for v in self.remaining_cities:
                x = input_func(self, v)

                Z = sum([exp(G[u][v]) for u in self.remaining_cities if v != u])
                p = [exp(G[u][v])/Z for u in self.remaining_cities if v != u]
                other_mu = [mu[u] for u in self.remaining_cities if v != u]

                l = sum([torch.matmul(a, b) for a, b in zip(p, other_mu)])

                next_mu[v] = F.relu(torch.matmul(self.W3, l) + torch.matmul(self.W4, x))

            mu = next_mu

        return mu

    def auction(self):
        pass

