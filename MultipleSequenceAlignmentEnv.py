import gymnasium as gym
import numpy as np
import pandas as pd

def getblosum62():
    return pd.read_csv('blosum62.csv', index_col=0)

class MultipleSequenceAlignmentEnv(gym.Env):
    def __init__(self, sequences):
        super(MultipleSequenceAlignmentEnv, self).__init__()
        self.sequences = sequences
        self.n_sequences = len(sequences)
        self.n_characters = 25 # 20 amino acids + 4 for DNA/RNA + 1 for gap
        
        self.max_length = max(len(sequence) for sequence in sequences) + 1 # Add 1 to account for gap insertions
        self.action_space = gym.spaces.MultiDiscrete([self.n_sequences, self.max_length])
        self.observation_shape = (self.n_sequences, self.max_length, self.n_characters)
        self.observation_space = gym.spaces.MultiDiscrete([self.n_characters] * self.n_sequences * self.max_length)
        self.blosum62 = getblosum62()
        self.state = None

    def reset(self):
        self.state = np.full((self.n_sequences, self.max_length), '-', dtype='<U1') # Fill the state with gap symbols
        for i, seq in enumerate(self.sequences):
            self.state[i, :len(seq)] = list(seq) # Add the original sequences to the state
        return self._get_observation()

    def step(self, action):
        seq_idx, pos = action
        self._insert_gap(seq_idx, pos)

        reward = self._calculate_reward()
        done = False
        info = {}

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        obs = np.zeros(self.observation_shape, dtype=int)
        for i, row in enumerate(self.state):
            for j, aa in enumerate(row):
                obs[i, j, self._aa_to_index(aa)] = 1
        return obs.flatten()

    def _aa_to_index(self, aa):
        aa = aa.upper()
        if aa == '-':
            return 20
        else:
            return ord(aa) - ord('A')

    def _insert_gap(self, seq_idx, pos):
        self.state[seq_idx, pos+1:] = np.roll(self.state[seq_idx, pos+1:], 1)
        self.state[seq_idx, pos] = '-'

    def _calculate_reward(self):
        reward = 0
        for col_idx in range(self.max_length):
            col = self.state[:, col_idx]
            col_score = 0

            for i in range(self.n_sequences):
                for j in range(i+1, self.n_sequences):
                    pair = tuple(sorted(col[i] + col[j]))
                    if pair in self.blosum62.index:
                        col_score += self.blosum62.loc[pair]# may not work as blosum62 is a dataframe

            reward += col_score

        return reward

    def print_alignment(self):
        for row in self.state:
            print(''.join(row))



if __name__ == "__main__":


    env = MultipleSequenceAlignmentEnv(['MCRIAGGRGTLLPLLAALLQA',
                                        'MSFPCKFVASFLLIFNVSSKGA',
                                        'MPGKMVVILGASNILWIMF'])
    obs = env.reset()

    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.print_alignment()
