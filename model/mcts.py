import numpy as np
import math

class Node:
    def __init__(self, environment, args, state, parent=None, action_taken=None):
        self.environment = environment
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.expandable_moves = environment.get_valid_moves(state)

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child
    
    def get_ucb(self, child):
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        # add +1 and /2 to q_value to make it in between 0 and 1, to be able to calculate probability
        # add 1 - () because the next move is the opponent
        return q_value + self.args['C']*math.sqrt(math.log(self.visit_count) / child.visit_count)
    
    def expand(self):
        action = np.random.choice(np.where(self.expandable_moves == 1)[0])
        self.expandable_moves[action] = 0

        child_state = self.state.copy()
        child_state = self.environment.get_next_state(child_state, action, 1)  # the child think that they are player 1
        child_state = self.environment.change_perspective(child_state, player=-1)

        child = Node(self.environment, self.args, child_state, self, action)
        self.children.append(child)

        return child
    
    def simulate(self):
        value, is_terminal = self.environment.get_value_and_terminated(self.state, self.action_taken)
        value = self.environment.get_opponent_value(value)

        if is_terminal:
            return value
        
        rollout_state = self.state.copy()
        rollout_player = 1
        while True:
            valid_moves = self.environment.get_valid_moves(rollout_state)
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state = self.environment.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.environment.get_value_and_terminated(rollout_state, action)
            if is_terminal:
                if rollout_player == -1:
                    value = self.environment.get_opponent_value(value)
                return value
            rollout_player = self.environment.get_opponent(rollout_player)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.environment.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)

class MCTS:
    def __init__(self, environment, args):
        self.environment = environment
        self.args = args

    def search(self, state):
        # define root
        root = Node(self.environment, self.args, state)

        for search in range(self.args['num_searches']):
            node = root
            # selection
            while node.is_fully_expanded():
                node = node.select()
            value, is_terminal = self.environment.get_value_and_terminated(node.state, node.action_taken)
            value = self.environment.get_opponent_value(value)
            
            if not is_terminal:
                # expansion
                node = node.expand()
                # simulation
                value = node.simulate()

            # backpropagation
            node.backpropagate(value)

        # return visit_counts
        action_probs = np.zeros(self.environment.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs