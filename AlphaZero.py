from MCTS import MCTS
import numpy as np

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

        steps = 50

        while steps > 0:
            action_probs = self.mcts.search(state)

            memory.append((state, action_probs))

            action = np.random.choice(self.MSA.nbr_sequences * self.MSA.max_length, p=action_probs)

            state = self.game.get_next_state(state, action)

            value = self.MSA.get_value(state)


        returnMemory = []
        for hist_state, hist_action_probs in memory:
            returnMemory.append(( self.MSA.get_encoded_state(hist_state), hist_action_probs))
        return returnMemory


    def train(self, memory):
        pass

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []

            self.model.eval()
            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                memory += self.selfPlay()

            self.model.train()
            for epoch in trange(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")