import dqn
import random

from dqn import ReplayMemory, Transition, Q_learning
from grid_game import GridWorld
from torch.autograd import Variable
from torch.nn import functional as F


import numpy as np
import torch.optim as optim
import torch

# Include the replay experience

def training_dqn(model,
               model_config,
               environ,
               environ_config,
               memory,
               memory_config,
               optimizer,
               optimizer_config,
               loss,
               loss_config,
               train_config):

    model = model(**model_config)
    memory = memory(**memory_config)
    optimizer = optimizer(model.parameters(), **optimizer_config)

    epsilon = train_config['epsilon']
    gamma = train_config['gamma']
    criterion = train_config['criterion']

    epochs = train_config['epochs']

    for i in range(epochs):
        # state = initGridPlayer()
        ev = environ(**environ_config)
        state = ev.state()

        game_status = 1
        step = 0

        #while game still in progress
        while game_status:

            Q = model(state)

            if (np.random.random() < epsilon): #choose random action
                action = random.choice(ev.possible_actions())
            else: #choose best action from Q(s,a) values
                action = np.argmax(Q.data)

            #Take action, observe new state S'
            new_state = ev.make_move(action)
            # print(state, new_state)
            step += 1

            #Observe reward
            reward = ev.reward(state, new_state)

            memory.push(state.data, action, new_state.data, reward)

            if (len(memory) < memory.capacity): #if buffer not filled, add to it
                state = new_state
                if reward != -1: #if reached terminal state, update game status
                    break
                else:
                    continue

            transitions = memory.sample()

            batch = Transition(*zip(*transitions))
            state_batch = Variable(torch.cat(batch.state))
            action_batch = Variable(torch.LongTensor(batch.action)).view(-1,1)
            new_state_batch = Variable(torch.cat(batch.new_state))
            reward_batch = Variable(torch.FloatTensor(batch.reward))
            non_final_mask = (reward_batch == -1)

            #Let's run our Q function on S to get Q values for all possible actions
            qval_batch = model(state_batch)

            # we only grad descent on the qval[action], leaving qval[not action] unchanged
            state_action_values = qval_batch.gather(1, action_batch)

            #Get max_Q(S',a)
            with torch.no_grad():
                newQ = model(new_state_batch)
            maxQ = newQ.max(1)[0]

            y = reward_batch
            y[non_final_mask] += gamma * maxQ[non_final_mask]
            y = y.view(-1,1)

            loss = criterion(state_action_values, y)

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.grad.data.clamp_(-1, 1)
            optimizer.step()

            print(f'Game #: {i}, loss : {loss}')

            state = new_state
            if reward != -1:
                game_status = 0
            if step >20:
                break

        if epsilon > 0.1:
            epsilon -= (1/epochs)

    return model

grid_solver = training_dqn(model = Q_learning,
                model_config = grid_model_config,
                environ = GridWorld,
                environ_config = grid_environ_config,
                memory = ReplayMemory,
                memory_config = grid_memory_config,
                optimizer = optim.RMSprop,
                optimizer_config = grid_optimizer_config,
                loss = torch.nn.MSELoss(),
                loss_config = {},
                train_config = grid_train_config)

def test_model(model, environ, environ_config):
    i = 0
    ev = environ(**environ_config)
    state = ev.state()

    print("Initial State:")
    print(ev)
    status = 1
    #while game still in progress
    while(status == 1):
        qval = model(state)
        print(qval)
        action = np.argmax(qval.data) #take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        state = ev.make_move(action)
        print(ev)
        reward = ev.reward(None, state)

        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 10):
            print("Game lost; too many moves.")
            break


grid_optimizer_config = {'lr' : 1e-2}

grid_model_config = {\
    'in_channels' : 64,
    'hidden_layers' : [150, 150],
    'out_channels' : 4,
    'unit' : dqn.hidden_unit,
    'activation' : F.relu, }

grid_memory_config = {\
    'capacity' : 80,
    'batch_size' : 40, }

grid_train_config = {\
    'epochs' : 1000,
    'gamma' : 0.9,
    'epsilon' : 1,
    'criterion' : torch.nn.MSELoss(),}

grid_environ_config = {\
    'game_type' : 1}


test_model(grid_solver, environ = GridWorld, environ_config = grid_environ_config)

from code import interact
interact(local = locals())
