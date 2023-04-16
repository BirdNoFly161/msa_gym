import numpy as np
import math
class Node:
    def __init__(self, MSA, args, state, parent=None, action_taken=None, legally_achieved=True):
        self.MSA = MSA
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.legally_achieved = legally_achieved
        self.children = []

        # expandable moves needed here or create it when needed ?
        # this is a 2d array that acts as a mask for legal moves
        self.expandable_moves = MSA.get_valid_moves(state)

        self.visit_count = 0
        self.value_sum = 0

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

    def expand(self):

        # choose an action from legal actions (might need to omit even more illegal ones (whether you can slide or not) )
        rows, cols = np.where(self.expandable_moves == 1)
        possible_actions = list(zip(rows, cols))
        action = possible_actions[ np.random.choice( len(possible_actions )) ]
        self.expandable_moves[action] = 0

        child_state = list(self.state).copy()
        sequence, column = action
        distance = 0
        if( column % 2 == 1):
            distance = 1
        else:
            distance = -1
        column = int(column / 2)
        child_state, legally_achieved = self.MSA.get_next_state(child_state, sequence, column, distance )

        child = Node(MSA=self.MSA, args=self.args, state=child_state,parent = self, action_taken=action, legally_achieved=legally_achieved)
        self.children.append(child)

        return child


    def simulate(self):

        steps = 10

        if(not self.legally_achieved):
            return -np.inf
        else:

            rollout_state = self.state.copy()

            while steps > 0:

                rows, cols = np.where(self.MSA.get_valid_moves(rollout_state) == 1)
                possible_actions = list(zip(rows, cols))
                action = possible_actions[np.random.choice(len(possible_actions))]

                sequence, column = action
                distance = 0
                if (column % 2 == 1):
                    distance = 1
                else:
                    distance = -1
                column = int(column / 2)
                rollout_state, legally_achieved = self.MSA.get_next_state(rollout_state, sequence, column, distance)
                steps -= 1

            value = self.MSA.get_value(rollout_state)
            return value

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count +=1

        if (not np.isinf(value)):
            if self.parent is not None:
                self.parent.backpropagate(value)







class MCTS:
    def __init__(self, MSA, args):
        self.MSA = MSA
        self.args = args

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


            node = node.expand()
            value = node.simulate()
            node.backpropagate(value)

        action_probs = np.zeros(self.MSA.action_size)

        for child in root.children:
            action_probs[child.action_taken] = child.visit_count

        action_probs /= np.sum(action_probs)
        return action_probs

