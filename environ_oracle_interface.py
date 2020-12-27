from abc import ABC, abstractmethod

class Action(ABC):

    @abstractmethod
    def apply_to_environ(self, environ):
        pass

class Environment(ABC):

    @abstractmethod
    def state(self):
        pass

class Oracle(ABC):

    @abstractmethod
    def decide_action(self):
        pass

if __name__ == '__main__':
    """Interface between environment and oracle.

    Overall flow shall be as following;

    cur_state = environ.state()
    next_action = oracle.decide_action(cur_state)
    """
    pass

