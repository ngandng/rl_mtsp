import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions
    
class DQNAgent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size = 100000, eps_end = 0.01, eps_dec = 5e-5, T = 1000):
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size                        # memory size
        self.batch_size = batch_size
        self.mem_cntr = 0                                   # memory counter: keep track of the position of the first available memory

        self.target_update_cntr = 0
        self.T = T                                          # period of updating target network

        # evaluation network
        self.policy_net = DeepQNetwork(self.lr, input_dims=input_dims, fc1_dims=256,
                                   fc2_dims=256, n_actions=n_actions)
        
        self.target_net = DeepQNetwork(self.lr, input_dims=input_dims, fc1_dims=256,
                                   fc2_dims=256, n_actions=n_actions)                       # note: target network has to be defined with same parameters as policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # storing memory
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)    # keep track of new state the agent accounted
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.termial_memory = np.zeros(self.mem_size, dtype=bool)                           # the done flag

    def store_transition(self, state, action, reward, new_state, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.termial_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(np.array([observation])).to(self.policy_net.device)  # send the observation tensor to device
            actions = self.policy_net.forward(state=state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def extract_greedy_action(self, observation):
        state = T.tensor(np.array([observation])).to(self.policy_net.device)  # send the observation tensor to device
        actions = self.policy_net.forward(state=state)
        action = T.argmax(actions).item()
        return action
    
    def update_target_net(self):
        if self.target_update_cntr % self.T == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def learn(self):
        # if our memory is just zero, just play randomly before learning
        if self.mem_cntr < self.batch_size:
            return
        
        # first, zero the gradient of our optimizer
        self.policy_net.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)     # select up to the last filled memory
        
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)    # to perform the proper array slicing

        # sample replay experience
        state_batch = T.tensor(self.state_memory[batch]).to(self.policy_net.device) # conver agent memory into pytorch tensor
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.policy_net.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.policy_net.device)
        terminal_batch = T.tensor(self.termial_memory[batch]).to(self.policy_net.device)

        action_batch = self.action_memory[batch]

        # get the predicted Q value
        q_pred = self.policy_net.forward(state_batch)[batch_index, action_batch]    # only need the value of actions we actually took
        
        # calculate target Q value
        # q_next = self.target_net.forward(new_state_batch)
        q_next = self.policy_net.forward(new_state_batch)
        q_next[terminal_batch] = 0.0                                            # the value of terminal states are 0
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]          # the max function return (value, index)

        loss = self.policy_net.loss(q_target, q_pred).to(self.policy_net.device)
        loss.backward()
        self.policy_net.optimizer.step()

        # epsilon decay
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon>self.eps_min else self.eps_min
        
        # update target network periodically 
        self.target_update_cntr += 1
        self.update_target_net()
        

