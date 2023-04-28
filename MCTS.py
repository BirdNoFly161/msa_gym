import numpy as np
import math
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Node:
    def __init__(self, MSA, args, state, parent=None, action_taken=None, prior=0):
        self.MSA = MSA
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        # this is a 2d array that acts as a mask for legal moves
        self.expandable_moves = MSA.legal_actions_mask.copy()

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def get_ucb(self, child):
        if(child.visit_count == 0):
            q_value = 0
        else:
            q_value = child.value_sum / child.visit_count

        return q_value + self.args['C']*math.sqrt(self.visit_count / (child.visit_count + 1 )) * child.prior

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

        for (x, y), probability in np.ndenumerate(policy):
            if probability > 0:
                child_state = self.MSA.get_next_state(self.state, (x,y))
                child = Node(MSA=self.MSA, args=self.args, state=child_state, parent=self, action_taken= (x,y), prior= probability)
                self.children.append(child)

        return child




    def backpropagate(self, value):
        self.value_sum += value
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
        self.model = model


    @torch.no_grad()
    def search(self, state):
        root = Node(self.MSA, self.args, state)

        for search in range(self.args['num_searches']):

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
            value = value.item()


            node = node.expand(policy)
            node.backpropagate(value)

        action_probs = np.zeros((self.MSA.nbr_sequences, self.MSA.max_length))

        for child in root.children:
            action_probs[child.action_taken] = child.visit_count

        action_probs /= np.sum(action_probs)
        action_probs = action_probs.flatten()
        return action_probs

