import numpy as np
import math
import torch
class Node:
    def __init__(self, MSA, args, state, parent=None, action_taken=None, prior= None):
        self.MSA = MSA
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children = []

        # this is a 2d array that acts as a mask for legal moves
        self.expandable_moves = MSA.legal_actions_mask.copy()

        self.visit_count = 0
        self.value_sum = -np.inf

    def is_fully_expanded(self):
        rows, cols = np.where(self.expandable_moves == 1)
        possible_actions = list(zip(rows, cols))
        return ( (len(possible_actions) == 0) and (len(self.children) > 0) )

    def get_ucb(self, child):
        if(child.visit_count == 0):
            q_value = 0
        else:
            q_value = child.value_sum / child.visit_count

        return q_value + self.args['C']*math.sqrt(math.log(self.visit_count) / child.visit_count)

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_ucb = ucb
                best_child = child

        return best_child

    def expand(self, policy):

        rows, cols = np.where(self.expandable_moves == 1)
        possible_actions = list(zip(rows, cols))
        action = possible_actions[ np.random.choice( len(possible_actions )) ]
        self.expandable_moves[action] = 0


        child_state = self.MSA.get_next_state(self.state, action)

        child = Node(MSA=self.MSA, args=self.args, state=child_state,parent = self, action_taken=action, prior= policy[action])
        self.children.append(child)

        return child


    def simulate(self):

        steps = 1

        rollout_state = {
            'Sequences': self.state['Sequences'].copy(),
            'Residue_positions': self.state['Residue_positions'].copy()
        }

        while steps > 0:

            action = tuple(self.MSA.action_space.sample())
            sequence, column = action
            rollout_state = self.MSA.get_next_state(rollout_state, action)
            steps -= 1

            value = self.MSA.get_value(rollout_state)
        return value

    def backpropagate(self, value):
        if value > self.value_sum:
            self.value_sum = value
        #self.value_sum += value
        self.visit_count +=1

        if (not np.isinf(value)):
            if self.parent is not None:
                self.parent.backpropagate(value)

        else:
            print('inf value encountered')







class MCTS:
    def __init__(self, MSA, args, model):
        self.MSA = MSA
        self.args = args
        self.model=model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.MSA, self.args, state)

        for search in range(self.args['num_searches']):

            if(search % 100 ==0):
                print(search)
            node = root
            #breakpoint
            brp=0
            while node.is_fully_expanded():
                node = node.select()


            encoded_state = self.MSA.get_encoded_state(state)
            tensor_state = torch.tensor(encoded_state, device = self.model.device).unsqueeze(0)

            policy, value = self.model(tensor_state)

            policy = torch.softmax(policy, axis= 1).squeeze(0).cpu()
            policy = policy.reshape(self.MSA.nbr_sequences, self.MSA.max_length)

            policy *= self.MSA.legal_actions_mask
            policy /= torch.sum(policy)

            node = node.expand(policy * 0.00001)
            value = node.simulate()
            node.backpropagate(value)

        action_probs = np.zeros((self.MSA.nbr_sequences, self.MSA.max_length))

        for child in root.children:
            action_probs[child.action_taken] = child.visit_count

        action_probs /= np.sum(action_probs)
        action_probs = action_probs.flatten()
        return action_probs

