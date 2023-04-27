from MCTS import MCTS
import numpy as np
import torch
import random
import torch.nn.functional as F
from tqdm import tqdm
class AlphaZero:
    def __init__(self, model, optimizer, MSA, args):
        self.model = model
        self.optimizer = optimizer
        self.MSA = MSA
        self.args = args
        self.mcts = MCTS(MSA, args, model)

    def selfPlay(self):
        memory = []
        state = self.MSA.reset()

        steps = 20

        while steps > 0:
            action_probs = self.mcts.search(state)

            action = np.random.choice(self.MSA.nbr_sequences * self.MSA.max_length, p=action_probs)
            action = np.unravel_index(action, (self.MSA.nbr_sequences, self.MSA.max_length))
            state = self.MSA.get_next_state(state, action)


            value = self.MSA.get_value(state)
            memory.append((state, action_probs, value))
            steps -= 1


        return memory



    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            states, policy_targets, value_targets = zip(*sample)

            policy_targets, value_targets = np.array(policy_targets), np.array(value_targets)

            policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
            value_targets = torch.tensor(value_targets, dtype=torch.float32)

            states = list(states)
            for idx,state in enumerate(states):
                states[idx] = self.MSA.get_encoded_state(state)
            states = np.array(states)
            states = torch.tensor(states)

            out_policy, out_value = self.model(states)





            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()  # change to self.optimizer
            loss.backward()
            self.optimizer.step()  # change to self.optimizer

    def learn(self):
        for iteration in tqdm(range(self.args['num_iterations'])):
            memory = []

            self.model.eval()
            for selfPlay_iteration in tqdm(range(self.args['num_selfPlay_iterations'])):
                print('self play iteration :', selfPlay_iteration)
                memory += self.selfPlay()

            self.model.train()
            for epoch in range(self.args['num_epochs']):
                print('epoch: ', epoch)
                self.train(memory)


            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")