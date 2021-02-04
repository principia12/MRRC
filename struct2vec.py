import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
from pprint import pprint
from math import exp
from random import sample

class Struct2vec(nn.Module):
    def __init__(self, mtsp_instance):
        self.mtsp_instance = mtsp_instance
        # self.action = action
        self.config = mtsp_instance.config
        N = self.config.N
        M = self.config.M
        tau = self.config.tau

        self.W1, self.W2 = torch.rand(1, N), torch.rand(N, 3)

        self.W3_A1, self.W3_A2 = torch.rand(M, M+1), torch.rand(M, M+1)
        self.W4_A1, self.W4_A2 = torch.rand(M, 1), torch.rand(M, 1)

        self.W3_B = torch.rand(M, M)
        self.W4_B = torch.rand(M, 2*M)

        self.W5 = torch.rand(1, M) # W8
        self.W6 = torch.rand(1, M) # W9

        self.W7 = torch.rand(1, M) # W10

    def layer_A1(self):
        G = defaultdict(lambda :{})
        N = self.config.N
        M = self.config.M
        tau = self.config.tau
        mtsp_instance = self.mtsp_instance

        graph = mtsp_instance.graph
        depot = mtsp_instance.depot

        for i, v in enumerate(graph.V):
            for j, u in enumerate(graph.V):
                # append g_ij
                vu_dist = graph[v][u] / 1000
                xv_dist = graph[depot][v] / 1000
                xu_dist = graph[depot][u] / 1000

                a = torch.FloatTensor([vu_dist, xv_dist, xu_dist])
                G[v][u] = torch.matmul(self.W1, F.relu(torch.matmul(self.W2, a)))
                # print(G[v][u])

        T = self.config.T

        # random mu initialization
        mu = {}
        next_mu = {}

        for v in mtsp_instance.remaining_cities:
            mu[v] = torch.rand(M, 1)
            next_mu[v] = None

        dist_from_robot = mtsp_instance.distance_from_robot()

        # main loop for struct2vec
        for t in range(T):
            for v in mtsp_instance.remaining_cities:

                Z = sum([exp(G[u][v]/tau) for u in mtsp_instance.remaining_cities if v != u])
                p = [exp(G[u][v]/tau)/Z for u in mtsp_instance.remaining_cities if v != u] # softmax of G given tau

                l_not_weighted = [torch.cat(\
                    (graph[u][v] * F.relu(torch.matmul(self.W5, mu[v])), mu[u])) \
                        for u in mtsp_instance.remaining_cities if u != v]
                # print(l_not_weighted[0].shape, 123213)
                l = sum([p_uv * l_uv for p_uv, l_uv in zip(p, l_not_weighted)])


                a = torch.matmul(self.W3_A1, l)
                next_mu[v] = F.relu(torch.matmul(self.W3_A1, l) \
                    + torch.matmul(self.W4_A1,
                        torch.FloatTensor([[dist_from_robot[v]]])))


            mu = next_mu

        return mu

    def layer_A2(self):
        G = defaultdict(lambda :{})
        N = self.config.N
        M = self.config.M
        tau = self.config.tau
        mtsp_instance = self.mtsp_instance

        graph = mtsp_instance.graph
        depot = mtsp_instance.depot

        for i, v in enumerate(graph.V):
            for j, u in enumerate(graph.V):
                # append g_ij
                vu_dist = graph[v][u]
                xv_dist = graph[depot][v]
                xu_dist = graph[depot][u]

                a = torch.FloatTensor([vu_dist, xv_dist, xu_dist])
                G[v][u] = torch.matmul(self.W1, F.relu(torch.matmul(self.W2, a)))
                # print(G[v][u])

        T = self.config.T

        # random mu initialization
        mu = {}
        next_mu = {}

        for v in mtsp_instance.remaining_cities:
            mu[v] = torch.rand(M, 1)
            next_mu[v] = None

        dist_from_depot = mtsp_instance.distance_from_depot()

        # main loop for struct2vec
        for t in range(T):
            for v in mtsp_instance.remaining_cities:

                Z = sum([exp(G[u][v]/tau) for u in mtsp_instance.remaining_cities if v != u])
                p = [exp(G[u][v]/tau)/Z for u in mtsp_instance.remaining_cities if v != u] # softmax of G given tau

                l_not_weighted = [torch.cat(\
                    (graph[u][v] * F.relu(torch.matmul(self.W6, mu[v])), mu[u])) \
                        for u in mtsp_instance.remaining_cities if u != v]

                l = sum([p_uv * l_uv for p_uv, l_uv in zip(p, l_not_weighted)])

                next_mu[v] = F.relu(torch.matmul(self.W3_A2, l) \
                    + torch.matmul(self.W4_A2,
                        torch.FloatTensor([[dist_from_depot[v]]])))

            mu = next_mu

        return mu

    def forward(self):

        A1 = self.layer_A1()
        A2 = self.layer_A2()

        layer_B_input = {}

        for v in self.mtsp_instance.remaining_cities:
            a = A1[v]
            b = A2[v]
            layer_B_input[v] = torch.cat((a, b))

        G = defaultdict(lambda :{})
        N = self.config.N
        M = self.config.M
        tau = self.config.tau
        mtsp_instance = self.mtsp_instance

        graph = mtsp_instance.graph
        depot = mtsp_instance.depot

        for i, v in enumerate(graph.V):
            for j, u in enumerate(graph.V):
                # append g_ij
                vu_dist = graph[v][u] / 1000
                xv_dist = graph[depot][v] / 1000
                xu_dist = graph[depot][u] / 1000

                a = torch.FloatTensor([vu_dist, xv_dist, xu_dist])
                G[v][u] = torch.matmul(self.W1, F.relu(torch.matmul(self.W2, a)))
                # print(G[v][u])

        T = self.config.T

        # random mu initialization
        mu = {}
        next_mu = {}

        for v in mtsp_instance.remaining_cities:
            mu[v] = torch.rand(M, 1)
            next_mu[v] = None

        # main loop for struct2vec
        for t in range(T):
            for v in mtsp_instance.remaining_cities:

                Z = sum([exp(G[u][v]/tau) for u in mtsp_instance.remaining_cities if v != u])
                p = [exp(G[u][v]/tau)/Z for u in mtsp_instance.remaining_cities if v != u] # softmax of G given tau

                l_not_weighted = [mu[u] for u in mtsp_instance.remaining_cities if u != v]

                l = sum([p_uv * l_uv for p_uv, l_uv in zip(p, l_not_weighted)])

                next_mu[v] = F.relu(\
                    torch.matmul(self.W3_B, l) \
                    + torch.matmul(self.W4_B, layer_B_input[v]))
                # print(layer_B_input[v].shape, 123)
            mu = next_mu

        return F.relu(torch.matmul(self.W7, sum([mu[v] for v in mtsp_instance.remaining_cities])))

    def auction(self):
        m = self.mtsp_instance

        chosen_cities = sample(m.remaining_cities, len(m.robots))

        return list(zip(m.robots, chosen_cities))


    def Q(self, s, a):

        with torch.no_grad():
            robot, next_city = a
            if len(s.remaining_cities) == 1 and s.remaining_cities[0] == next_city:
                return -1
            elif robot.assigned_city is None: # robot shall be in the depot
                last_city = robot.location_history[-1]
                robot.assigned_city = next_city
                robot.remaining_distance = s.graph[last_city][next_city]
                q = self.forward()
                robot.assigned_city = None
                robot.remaining_distance = -1
                return q
            else:
                raise ValueError
        # return Q

class DummyBrain:
    def __init__(self, mtsp_instance):
        self.mtsp_instance = mtsp_instance

    def initial_assignment(self):

        m = self.mtsp_instance

        chosen_cities = sample(m.remaining_cities, len(m.robots))

        return list(zip(m.robots, chosen_cities))

    def state(self):
        pass

    def Q(self, s, a):
        """Calculate Q(s, a).

        s is a state of current mtsp object.
        action should be tuple of robot-city assignment

        """

        r, c = a
        if c == min(s.remaining_cities,
                        key = lambda c:s.graph[r.location_history[-1]][c]):
            return 0
        else:
            return 1
        return Q