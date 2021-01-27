import os

from environ_oracle_interface import Environment
from graph import Edge, Graph

from collections import namedtuple
from pprint import pformat, pprint
from math import sqrt

from struct2vec import

class Robot:
    def __init__(self, robot_id, initial_position):
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
        self.city_id = city_id
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
        assert depot in graph.V

        self.graph = Graph.from_nodes(cities,
                        lambda x, y : Edge(x, y, x.loc - y.loc))
        self.depot = depot

        self.remaining_cities = [v for v in graph.V if v != depot]
        self.available_robots = self.robots
        self.epoch = 0
        self.robots = robots

        self.config = DummyCls(**config)

    @staticmethod
    def _distance_from_robot(self):
        pass

    @staticmethod
    def _distance_from_depot(self):
        pass

    def state(self):
        pass

    def possible_actions(self):
        if len(self.available_robots) > 2:
            return self.auction()
        else:
            available_robot = self.available_robots[0]
            src = available_robot.location_history[-1]

            return [MTSP.Action(available_robot, src, dest) for dest in self.remaining_cities]

    def auction(self):
        return []

    def make_move(self, action):
        robot, src, dest = action
        assert src == robot.location_history[-1]
        assert robot.is_available
        assert dest in self.remaining_cities

        #
        robot.location_history.append(dest)
        robot.is_available = False
        robot.assigned_city = dest
        robot.remaining_distance = self.graph[src][dest]

        self.remaining_cities.remove(dest)

        min_distance, next_arriving_robot = min([r.remaining_distance, r \
                for r in self.robots if r.remaining_distance > 0])

        for r in self.robots:
            if r.remaining_distance > 0:
                r.remaining_distance -= min_distance



        return self.state()




    @staticmethod
    def _from_file(file_path):
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
                locations[0],
                Robot.generate_robots(i, locations[0])))

        return instances

if __name__ == '__main__':
    instances = MTSP._from_file(os.path.join('data', 'berlin52.tsp'))

