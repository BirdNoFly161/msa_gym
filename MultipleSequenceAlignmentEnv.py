import gymnasium as gym
import numpy as np
import pandas as pd
# this is a very simple environment that only inserts gaps
# the reward is the sum of the blosum62 scores of all pairs of sequences
# the observation is a one-hot encoding of the sequences and the gaps
# the action is the index of the sequence and the position of the gap
# this not working yet
# but should be rather efficient when it is
def getblosum62():
    return pd.read_csv('blosum62.csv', index_col=0)

class MultipleSequenceAlignmentEnv(gym.Env):
    def __init__(self, sequences):
        super(MultipleSequenceAlignmentEnv, self).__init__()
        self.sequences = sequences
        self.n_sequences = len(sequences)
        self.max_length = max(len(sequence) for sequence in sequences)
        self.action_space = gym.spaces.MultiDiscrete([self.n_sequences, self.max_length])
        self.observation_space = gym.spaces.MultiDiscrete([20] * self.n_sequences * self.max_length)
        self.blosum62 = getblosum62()
        self.state = None

    def reset(self):
        self.state = np.zeros((self.n_sequences, self.max_length), dtype=int)
        return self._get_observation()

    def step(self, action):
        seq_idx, pos = action
        self._insert_gap(seq_idx, pos)

        reward = self._calculate_reward()
        done = False
        info = {}

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        obs = np.zeros((self.n_sequences, self.max_length, 21), dtype=int)
        for i, seq in enumerate(self.sequences):
            for j, aa in enumerate(seq):
                obs[i, j, self._aa_to_index(aa)] = 1
        obs[self.state == 1] = 0
        obs[self.state == 2] = 1
        return obs.flatten()

    def _aa_to_index(self, aa):
        aa = aa.upper()
        if aa == '-':
            return 0
        else:
            return ord(aa) - ord('A') + 1

    def _insert_gap(self, seq_idx, pos):
        self.state[seq_idx, pos+1:] = np.roll(self.state[seq_idx, pos+1:], 1)
        self.state[seq_idx, pos] = 0

    def _calculate_reward(self):
        reward = 0
        for col_idx in range(self.max_length):
            col = self.state[:, col_idx]
            col_score = 0

            for i in range(self.n_sequences):
                for j in range(i+1, self.n_sequences):
                    pair = tuple(sorted((col[i], col[j])))
                    col_score += self.blosum62.get(pair, 0)

            reward += col_score

        return reward


if __name__ == "__main__":


    env = MultipleSequenceAlignmentEnv(['ACDEFGHIKLMNPQRSTVWY', 'ACDEFGHIKLMNPQRSTVWY', 'ACDEFGHIKLMNPQRSTVWY'])
    obs = env.reset()

    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
