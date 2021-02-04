import os

from environ_oracle_interface import Environment
from graph import Edge, Graph

from collections import namedtuple
from pprint import pformat, pprint
from math import sqrt

from struct2vec import Struct2vec

class Robot:
    def __init__(self, robot_id, initial_position):
        """Wrapper class for robot.

        Attributes:
            robot_id: int
                id of robot.
            location_history: listof int
                Recording previous positions of robot. The last element of the location_history should be the current position of the robot in any time.
            is_available: bool
                If robot is not assigned to any of the tasks, is_available is True.
                Else, it must be False.
            assigned_city: None or int
                If None,
                    1) The robot is in its initial state
                    2) The robot have no more task to be assigned.
                If assigned_city is None, is_available must be True.

                When robot have arrived to the task, its next location shall be determined at the moment.
            remaining_distance: float
                Remaining distance to the assigned_city.
        """
        self.robot_id = robot_id
        self.location_history = [initial_position]
        self.is_available = True
        self.assigned_city = None
        self.remaining_distance = -1

    @staticmethod
    def generate_robots(n, loc):
        robots = []
        for i in range(n):
            robots.append(Robot(i, loc))
        return robots

class Coordinate:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __sub__(self, other):
        return sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __str__(self):
        return f'({self.x}, {self.y})'

    @staticmethod
    def from_str(text):
        return Coordinate(*[float(e) for e in text.split(',').strip('(').strip(')')])

class City:
    def __init__(self, city_id, x, y):
        self.city_id = int(city_id)
        self.x = x
        self.y = y
        self.loc = Coordinate(x, y)

class DummyCls:
    def __init__(self, **kargs):
        for k, v in kargs.items():
            exec(f'self.{k} = v')

class MTSP(Environment):
    Action = namedtuple('action', ['robot', 'src', 'dest'])

    def __init__(self, cities, depot, robots, **config ):

        V = []
        E = []

        for c in cities:
            V.append(c.city_id)
            for d in cities:
                E.append(Edge(c.city_id, d.city_id, (c.loc - d.loc) / 1000))

        self.graph = Graph(V, E)
        self.depot = depot
        assert depot in self.graph.V, (depot, self.graph.V)

        self.remaining_cities = [v for v in self.graph.V if v != depot]
        self.available_robots = robots
        self.epoch = 0
        self.robots = robots

        self.config = DummyCls(**config)

    def distance_from_robot(self):
        res = {}

        for v in self.graph.V:
            dists = []
            for r in self.robots:
                if r.assigned_city is not None:
                    dists.append(self.graph[r.assigned_city][v] + r.remaining_distance)
                else:
                    dists.append(0)
            res[v] = min(dists)

        return res

    def distance_from_depot(self):
        return self.graph[self.depot]

    def possible_actions(self):
        if len(self.available_robots) > 2:
            return self.auction()
        else:
            available_robot = self.available_robots[0]
            src = available_robot.location_history[-1]

            return [MTSP.Action(available_robot, src, dest) for dest in self.remaining_cities]

    def auction(self):
        return []

    def state(self):
        pass

    def make_move(self, actions):
        for action in actions:
            robot, dest = action
            assert robot.is_available
            assert dest in self.remaining_cities

            #
            src = robot.location_history[-1]
            robot.location_history.append(dest)
            robot.is_available = False
            robot.assigned_city = dest
            try:
                robot.remaining_distance = self.graph[src][dest]
            except KeyError:
                # pprint(list(self.graph.adj_list.keys()))
                from code import interact
                assert False, interact(local = locals())

            self.remaining_cities.remove(dest)

        min_distance, next_arriving_robot = min([(r.remaining_distance, r) \
                for r in self.robots if r.remaining_distance > 0], key = lambda x:x[0])

        for r in self.robots:
            if r.remaining_distance > 0:
                r.remaining_distance -= min_distance

        next_robots = []

        for r in self.robots:
            if r.remaining_distance == 0:
                next_robots.append(r)
                r.is_available = True
                r.assigned_city = None
                r.remaining_distance = -1

        return min_distance, next_robots

    @staticmethod
    def _from_file(file_path, **config):
        with open(file_path, 'r') as f:
            locations = []
            for line in f.readlines():
                if line.split()[0].isnumeric():
                    loc_id, x, y = line.split()
                    locations.append(City(loc_id, float(x), float(y)))

        instances = []
        for i in [2,3,5,7]:
            instances.append(MTSP(\
                locations,
                locations[0].city_id,
                Robot.generate_robots(i, locations[0].city_id),
                **config))

        return instances

if __name__ == '__main__':
    instances = MTSP._from_file(os.path.join('data', 'berlin52.tsp'),
                        N = 10,
                        M = 20,
                        tau = 10,
                        T = 1)

    from struct2vec import DummyBrain, Struct2vec
    DEBUG = False
    for m in instances[1:2]:
        from time import time
        begin = time()
        brain = DummyBrain(m)
        s = Struct2vec(m)
        t = Struct2vec(m)

        actions = brain.initial_assignment()
        actions = s.auction()

        cost = 0
        pprint([(r.robot_id, c) for r, c in actions])

        cost_per_robot = [0 for v in m.robots]
        trip_per_robot = [[] for v in m.robots]

        while m.remaining_cities != []:
            min_distance, next_robots = m.make_move(actions)

            cost_per_robot[next_robots[0].robot_id] += min_distance
            trip_per_robot[next_robots[0].robot_id].append([next_robots[0].location_history[-1]])
            if DEBUG:
                print('===============')
                print(f'epoch took {min_distance} time')
                print(f'remaining cities : {m.remaining_cities}')

            if m.remaining_cities == []:
                r = next_robots[0]
                next_city = r.location_history[-1]
                print(f'{r.robot_id} : {r.location_history[-1]} --> {next_city}')
                trip_per_robot[next_robots[0].robot_id][-1].append(next_city)
                break

            cost += min_distance
            # expected_values = brain.q()
            expected_values = s.Q
            expected_noise_values = t.Q

            actions = []

            for r in next_robots:
                assert r.assigned_city is None
                vals = []
                for c in m.remaining_cities:
                    qval = brain.Q(m, (r,c))
                    s2v_q = s.Q(m, (r,c))
                    s2v_q_target = t.Q(m, (r,c))
                    vals.append(((r,c), qval))
                    if DEBUG:
                        print(f'action : {r.robot_id} to {c}')
                        print(f'exp : {qval} ')
                        print(f's2v : {s2v_q}')
                        print(f's2v_target : {s2v_q_target}')
                next_city = min(m.remaining_cities,
                            key = lambda c:s.Q(m, (r, c)))
                print(f'{r.robot_id} : {r.location_history[-1]} --> {next_city}')
                trip_per_robot[next_robots[0].robot_id][-1].append(next_city)

                actions.append((r, next_city))
        cost_per_robot = []
        print('============')
        for r_idx, trips in enumerate(trip_per_robot):
            print(r_idx)
            print(trips)
            for src, dest in trips:
                print(f'{src} --> {dest}, {m.graph[src][dest]}')
            print(sum([m.graph[s][d] for s, d in trips]))
            cost_per_robot.append(sum([m.graph[s][d] for s, d in trips]))
        print('============')
        print(max(cost_per_robot) * 1000)
        end = time()
        print(end - begin)
        # print(cost)

        # optimizer.step()

    # for m in instances:
        # brain = Struct2vec(
