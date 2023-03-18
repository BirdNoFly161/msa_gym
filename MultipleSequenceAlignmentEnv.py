import gymnasium as gym
import numpy as np
import pandas as pd

def getblosum62():
    return pd.read_csv('blosum62.csv', index_col=0)

class MultipleSequenceAlignmentEnv(gym.Env):
    alphabet = 'ARNDCEQGHILKMFPSTYWV' # 20 amino acids also works for DNA/RN
    onehot_transform = dict(zip(alphabet, range(len(alphabet))))
    weight_matrix = getblosum62()
    def __init__(self, sequences):
        super(MultipleSequenceAlignmentEnv, self).__init__()
        self.state = None
        self.sequences = sequences
        self.n_sequences = len(sequences)
        self.n_characters = len(self.alphabet) 
        self.max_length = max(len(sequence) for sequence in sequences)
        self.action_space = gym.spaces.MultiDiscrete([self.n_sequences, self.max_length])
        self.observation_space = gym.spaces.Dict({
            'one_hot_sequences' : gym.spaces.MultiBinary([self.n_sequences, self.max_length, self.n_characters]),
            'positions' : gym.spaces.MultiDiscrete([self.n_sequences, self.max_length])
            # profiles could be a good idea
            #'profiles' : gym.spaces.MultiBinary([self.max_length, self.n_characters])
        })
        
        

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
                obs[i, j, self.letter_2_index(aa)] = 1
        return obs.flatten()

    def letter_2_index(self, letter):
        print(letter)
        return self.alphabet.index(letter.upper())

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
                    if pair in self.weight_matrix.index:
                        col_score += self.weight_matrix.loc[pair]# may not work as blosum62 is a dataframe

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
    print(action)
    obs, reward, done, info = env.step(action)
    env.print_alignment()
