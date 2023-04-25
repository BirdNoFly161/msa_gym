import gymnasium as gym
import numpy as np
import pandas as pd
from itertools import combinations
from math import comb
# make numpy matrix print one line
np.set_printoptions(linewidth=200)


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
        reward = current_score - self.score
        self.score = current_score
        return reward # reward is the difference between the current score and the previous score
    
    def column_score(self):
        # the trick is to calculate the score of each column and then sum them without generating the string alignment 
        # this is much faster than generating the alignment and then calculating the score
        max_len = np.max(self.state) # length of the longest aligned sequence
        score = 0
        for col in range(1,max_len+1):
            all_basis_on_column = list(zip(*np.where(self.state == col)))
            #number_of_indels = self.n_sequences - len(all_basis_on_column[0]) # number of indels in the column
            for (i,j),(k,l) in combinations(all_basis_on_column, 2):
                score += self.weight_matrix.loc[self.sequences[i][j], self.sequences[k][l]]
            # to penalize them, we can add a penalty for each indel in the column
            # gap_penalty
            gap_penalty = -4 # this is the gap penalty in blosum62
            n_non_gaps = len(all_basis_on_column)
            # (N-m) * m + (N-m)! / (2! * (N-m-2)!) * gap_penalty
            # where N is self.n_sequences and m is n_non_gaps
            score += (self.n_sequences - n_non_gaps) * n_non_gaps + comb(self.n_sequences - n_non_gaps, 2) * gap_penalty

        
        return score
    

    def print_mat_string_alignment(self):
        # max_len = np.max(self.state)
        # for i, seq in enumerate(self.sequences):
        #     aligned_seq = ''
        #     for j, base in enumerate(seq):
        #         if self.state[i, j] > len(aligned_seq):
        #             aligned_seq += '-' * (self.state[i, j] - len(aligned_seq))
        #         aligned_seq += base
        #     aligned_seq += '-' * (max_len - len(aligned_seq))
        #     print(aligned_seq)
        align_mat = self.mat_string_alignment()
        for i in range(self.n_sequences):
            print(''.join(align_mat[i,:]))



    
    def mat_string_alignment(self):
        # length of the longest aligned sequence
        max_len = np.max(self.state)
        # make a matrix full of gaps
        alignment = np.full([self.n_sequences, max_len], '-')
        # fill the matrix with the aligned bases in the correct positions
        for i,seq in enumerate(self.sequences):
            for j,base in enumerate(seq):
                alignment[i, self.state[i,j]-1] = base
        return alignment
        




if __name__ == "__main__":

    env = MultipleSequenceAlignmentEnv(['FGKGKC',
       'FGKFGK',
       'GKGKC',
       'KFKC'])
    obs = env.reset()
    for _ in range (15):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(action, reward)
    print(env.state)
    env.print_mat_string_alignment()
    print(env.mat_string_alignment())
