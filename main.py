import dqn

from dqn import ReplayMemory, Transition, Q_learning
from torch.autograd import Variable
from grid_game import GridWorld
import torch.optim as optim
import torch

## Include the replay experience

epochs = 1000
gamma = 0.9 #since it may take several moves to goal, making gamma high
epsilon = 1
model = Q_learning(64, [150,150], 4, hidden_unit)
optimizer = optim.RMSprop(model.parameters(), lr = 1e-2)
# optimizer = optim.SGD(model.parameters(), lr = 0.1, momentum = 0)
criterion = torch.nn.MSELoss()
buffer = 80
batch_size = 40
memory = ReplayMemory(buffer)

class TrainConfig:
    def __init__(self, kargs):
        for k, v in kargs.items():
            exec(f'setattr(self, {k}, v)')

model_config = {\
    'in_channels' : 64,
    'hidden_layers' : [150, 150],
    'out_channels' : 4,
    'unit' : dqn.hidden_unit,
    'activation' : F.relu, }

memory_config = {\
    'buffer' : 80,
    'batch_size' : 40, }

grid_train_config = {\
    epochs = 1000,
    gamma = 0.9,
    epsilon = 1,
    optimizer = optim.RMSprop,
    lr = 1e-2,
    criterion = torch.nn.MSELoss(),

def training_dqn(model,
               model_config,
               environ,
               environ_config,
               memory,
               memory_config,
               train_config):

    model = model(**model_config)
    memory = memory(**memory_config)

    for i in range(train_config.epochs):
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
            step += 1

            #Observe reward
            reward = ev.reward(state, new_state)

            memory.push(state.data, action, new_state.data, reward)

            if (len(memory) < buffer): #if buffer not filled, add to it
                state = new_state
                if reward != -1: #if reached terminal state, update game status
                    break
                else:
                    continue

            transitions = memory.sample(BATCH_SIZE)

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
            # if reward == -1: #non-terminal state
                # update = (reward + (gamma * maxQ))
            # else: #terminal state
                # update = reward + 0*maxQ
            # y = reward_batch + (reward_batch == -1).float() * gamma *maxQ
            y = reward_batch
            y[non_final_mask] += gamma * maxQ[non_final_mask]
            y = y.view(-1,1)
            print("Game #: %s" % (i,), end='\r')
            loss = criterion(state_action_values, y)
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            for p in model.parameters():
                p.grad.data.clamp_(-1, 1)
            optimizer.step()
            state = new_state
            if reward != -1:
                status = 0
            if step >20:
                break
        if epsilon > 0.1:
            epsilon -= (1/epochs)



## Here is the test of AI
def testAlgo(init=0):
    i = 0
    if init==0:
        state = initGrid()
    elif init==1:
        state = initGridPlayer()
    elif init==2:
        state = initGridRand()

    print("Initial State:")
    print(dispGrid(state))
    status = 1
    #while game still in progress
    while(status == 1):
        v_state = Variable(torch.from_numpy(state))
        qval = model(v_state.view(64))
        print(qval)
        action = np.argmax(qval.data) #take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        state = makeMove(state, action)
        print(dispGrid(state))
        reward = getReward(state)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 10):
            print("Game lost; too many moves.")
            break

training_dqn(model = Q_learning(64, [150,150], 4, hidden_unit),
    )


testAlgo(init=1)
from code import interact
interact(local = locals())
