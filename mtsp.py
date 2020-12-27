import os

from environ_oracle_interface import Environment
from graph import Edge, Graph
from pprint import pformat, pprint
from math import sqrt

class Robot:
    def __init__(self, robot_id, location):
        self.robot_id = robot_id
        self.location = location
        self.is_available = True

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

class MSTP(Environment):
    def __init__(self, cities, depot, robots ):
        self.graph = Graph.from_nodes(cities,
                        lambda x, y : Edge(x, y, x.loc - y.loc))
        self.depot = depot
        self.epoch = 0
        self.robots = robots

    def state(self):
        return self.graph

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
            instances.append(MSTP(\
                locations,
                locations[0],
                Robot.generate_robots(i, locations[0])))

        return instances

if __name__ == '__main__':
    instances = MSTP._from_file(os.path.join('data', 'berlin52.tsp'))

