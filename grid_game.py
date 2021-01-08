import numpy as np
import torch

from pprint import pprint, pformat
from torch.autograd import Variable

from environ_oracle_interface import Environment

try: # when programming in company
    from devtools.debugger import debug_shell
except ImportError:
    from code import interact
    debug_shell = lambda : interact(local = locals())

def rand_pair(s,e):
    return np.random.randint(s,e), np.random.randint(s,e)

#finds an array in the "depth" dimension of the grid
def find_loc(state, obj):

    for i in range(0,4):
        for j in range(0,4):
            if (state[i,j] == obj).all():
                return i,j

#Initialize stationary grid, all items are placed deterministically
def init_grid():
    state = np.zeros((4,4,4))
    #place player
    state[0,1] = np.array([0,0,0,1])
    #place wall
    state[2,2] = np.array([0,0,1,0])
    #place pit
    state[1,1] = np.array([0,1,0,0])
    #place goal
    state[3,3] = np.array([1,0,0,0])
    return state

#Initialize player in random location, but keep wall, goal and pit stationary
def init_grid_player():
    state = np.zeros((4,4,4))
    #place player
    state[rand_pair(0,4)] = np.array([0,0,0,1])
    #place wall
    state[2,2] = np.array([0,0,1,0])
    #place pit
    state[1,1] = np.array([0,1,0,0])
    #place goal
    state[1,2] = np.array([1,0,0,0])

    a = find_loc(state, np.array([0,0,0,1])) #find grid position of player (agent)
    w = find_loc(state, np.array([0,0,1,0])) #find wall
    g = find_loc(state, np.array([1,0,0,0])) #find goal
    p = find_loc(state, np.array([0,1,0,0])) #find pit
    if (not a or not w or not g or not p):
        return init_grid_player()

    return state

#Initialize grid so that goal, pit, wall, player are all randomly placed
def init_grid_rand():
    state = np.zeros((4,4,4))
    #place player
    state[rand_pair(0,4)] = np.array([0,0,0,1])
    #place wall
    state[rand_pair(0,4)] = np.array([0,0,1,0])
    #place pit
    state[rand_pair(0,4)] = np.array([0,1,0,0])
    #place goal
    state[rand_pair(0,4)] = np.array([1,0,0,0])

    a = find_loc(state, np.array([0,0,0,1]))
    w = find_loc(state, np.array([0,0,1,0]))
    g = find_loc(state, np.array([1,0,0,0]))
    p = find_loc(state, np.array([0,1,0,0]))

    #If any of the "objects" are superimposed, just call the function again to re-place
    if (not a or not w or not g or not p):
        #print('Invalid grid. Rebuilding..')
        return init_grid_rand()

    return state

def make_move(state, action):
    #need to locate player in grid
    #need to determine what object (if any) is in the new grid spot the player is moving to
    player_loc = find_loc(state, np.array([0,0,0,1]))
    wall = find_loc(state, np.array([0,0,1,0]))
    goal = find_loc(state, np.array([1,0,0,0]))
    pit = find_loc(state, np.array([0,1,0,0]))
    # print(display_grid(state))
    state = np.zeros((4,4,4))

    actions = [[-1,0],[1,0],[0,-1],[0,1]]
    #e.g. up => (player row - 1, player column + 0)
    # up: 0, down : 1, left: 2 right: 3

    new_loc = [player_loc[0] + actions[action][0], player_loc[1] + actions[action][1]]


    new_loc[0] = max(0, new_loc[0])
    new_loc[1] = max(0, new_loc[1])
    new_loc[0] = min(3, new_loc[0])
    new_loc[1] = min(3, new_loc[1])

    new_loc = tuple(new_loc)

    # print(player_loc, new_loc, action)
    if (new_loc != wall):
        if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
            state[new_loc][3] = 1

    new_player_loc = find_loc(state, np.array([0,0,0,1]))
    if (not new_player_loc):
        state[player_loc] = np.array([0,0,0,1])
    #re-place pit
    state[pit][1] = 1
    #re-place wall
    state[wall][2] = 1
    #re-place goal
    state[goal][0] = 1

    return state

def get_loc(state, level):
    for i in range(0,4):
        for j in range(0,4):
            if (state[i,j][level] == 1):
                return i,j

def get_reward(state):
    player_loc = get_loc(state, 3)
    pit = get_loc(state, 1)
    goal = get_loc(state, 0)
    if (player_loc == pit):
        return -10
    elif (player_loc == goal):
        return 10
    else:
        return -1

def display_grid(state):
    grid = np.zeros((4,4), dtype= str)
    player_loc = find_loc(state, np.array([0,0,0,1]))
    wall = find_loc(state, np.array([0,0,1,0]))
    goal = find_loc(state, np.array([1,0,0,0]))
    pit = find_loc(state, np.array([0,1,0,0]))
    for i in range(0,4):
        for j in range(0,4):
            grid[i,j] = ' '

    if player_loc:
        grid[player_loc] = 'P' #player
    if wall:
        grid[wall] = 'W' #wall
    if goal:
        grid[goal] = '+' #goal
    if pit:
        grid[pit] = '-' #pit

    return grid


class GridWorld(Environment):
    STATE_INITIALIZER_DICT = {\
        0 : init_grid,
        1 : init_grid_player,
        2 : init_grid_rand, }

    def __init__(self, game_type = 0):
        self._state = GridWorld.STATE_INITIALIZER_DICT[game_type]()

    def state(self):
        return Variable(torch.from_numpy(self._state)).view(1, -1)

    def __str__(self):
        return pformat(display_grid(self._state))

    def possible_actions(self):
        return list(range(4))

    def make_move(self, action):
        self._state = make_move(self._state, action)
        # print(self)
        # print(action)
        return self.state()

    def reward(self, before_state, cur_state):
        return get_reward(cur_state.view(4,4,4))
