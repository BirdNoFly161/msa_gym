import gymnasium as gym
import numpy as np
import pandas as pd

def getblosum62():
    return pd.read_csv('blosum62.csv', index_col=0)

class MultipleSequenceAlignmentEnv(gym.Env):
    alphabet = 'ARNDCEQGHILKMFPSTYWV' # 20 amino acids also works for DNA/RN
    onehot = np.eye(len(alphabet)) # binary encoding of the alphabet
    onehot_transform = dict(zip(alphabet, range(len(alphabet)))) # dictionary to transform letters to indices

    weight_matrix = getblosum62()
    def __init__(self, sequences):
        super(MultipleSequenceAlignmentEnv, self).__init__()
        self.score = 0
        self.state = None
        self.one_hot_sequences = None
        self.sequences = sequences
        self.n_sequences = len(sequences)
        self.n_characters = len(self.alphabet) 
        self.max_length = max(len(sequence) for sequence in sequences)
        self.action_space = gym.spaces.MultiDiscrete([self.n_sequences, self.max_length])
        self.observation_space = gym.spaces.Dict({
            'one_hot_sequences' : gym.spaces.MultiBinary([self.n_sequences, self.max_length, self.n_characters]),
            'positions' : gym.spaces.MultiDiscrete([self.n_sequences, self.max_length]),
            # profile could be a good idea for the model to know the distribution of amino acids in each position 
            #'profile' : gym.spaces.MultiBinary([self.max_length, self.n_characters])
        })
        
        

    def reset(self):
        self.state = np.zeros([self.n_sequences, self.max_length], dtype=np.int_) # watch out for overflow
        self.one_hot_sequences = np.zeros([self.n_sequences, self.max_length, self.n_characters], dtype=np.int8) # too big for binary
        for i, sequence in enumerate(self.sequences):
            # one hot encoding of the sequences
            self.one_hot_sequences[i, :len(sequence), :] = np.array([self.onehot[self.onehot_transform[letter]] for letter in sequence])
            # state is just the indices of each amino acide which is 1,2,3,4...len(sequence)+1
            self.state[i, :len(sequence)] = np.arange(1, len(sequence)+1)
        self.score = self.column_score()
        return self._get_observation()
            


    def step(self, action):
        seq_idx, pos = action
        self._insert_gap(seq_idx, pos)

        reward = self._calculate_reward()
        self.score += reward
        done = False # MSA is an episodic task but we don't know when it ends
        info = {}

        return self._get_observation(), reward, done, info

    def _get_observation(self):
        return {
            'one_hot_sequences' : self.one_hot_sequences,
            'positions' : self.state,
            #'profile' : self.profile
        }


    def _insert_gap(self, seq_idx, pos):
        # inserting a gap in the sequence seq_idx at position pos
        self.state[seq_idx, pos:len(self.sequences[seq_idx])] += 1

    def _calculate_reward(self):
        current_score = self.column_score()
        return current_score - self.score # reward is the difference between the current score and the previous score
    
    def column_score(self):
        # the trick is to calculate the score of each column and then sum them without generating the string alignment 
        # this is much faster than generating the alignment and then calculating the score
        return 0

    def print_alignment(self):
        # state is the indices of the amino acids in each sequence
        alignment_max_len = np.max(self.state) - 1
        for i,seq in enumerate(self.sequences):
            accumulated_indels = 0
            for j,base in enumerate(seq):
                number_of_indels = self.state[i,j] - j - 1 - accumulated_indels
                accumulated_indels = accumulated_indels + number_of_indels
                print('_'*number_of_indels, base, end='',sep='')
            print('_'*(alignment_max_len - len(seq)+1))



if __name__ == "__main__":


    env = MultipleSequenceAlignmentEnv(['MCRIAGGRGTLLPLLAALLQA',
                                        'MSFPCKFVASFLLIFNVSSKGA',
                                        'MPGKMVVILGASNILWIMF'])
    obs = env.reset()

    action = env.action_space.sample()
    print(action)
    obs, reward, done, info = env.step(action)
    env.print_alignment()
