import gymnasium as gym
import numpy as np
import blosum as bl
import re
import itertools
from gymnasium import spaces
import skbio.io
from skbio import Protein, DNA


class MsaGrid(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, render_mode= None, sequences_filename= None, sequence_constructor= None):
        sequences = list(skbio.read(sequences_filename, format='fasta'))

        if sequence_constructor == 'protein':
            self.sequence_constructor = Protein
        if sequence_constructor == 'dna':
            self.sequence_constructor = DNA

        self.initial_sequences = list(map(self.sequence_constructor, sequences))
        self.max_length = max(len(sequence) for sequence in self.initial_sequences)
        self.nbr_sequences = len(self.initial_sequences)

        padded_sequences = list()
        for idx, sequence in enumerate(self.initial_sequences):

            if len(sequence) < self.max_length:
                padded_sequences.append(self.sequence_constructor.concat(
                    [sequence, self.sequence_constructor('-' * (self.max_length - len(sequence)))])
                )
            else:
                padded_sequences.append((sequence))

        self.initial_padded_sequences = padded_sequences

        self.observation_space = spaces.Dict({
            'Sequences': spaces.Sequence(
                spaces.Text(charset=self.sequence_constructor.alphabet, max_length=self.max_length)),

            'Residue_positions': spaces.MultiDiscrete( [self.nbr_sequences, self.max_length] )

        })

        self.action_space = spaces.MultiDiscrete( [self.nbr_sequences, self.max_length] )
        self.legal_actions_mask = np.ones( (self.nbr_sequences, self.max_length) )

        # mask out illegal move, ie make sure sequences < max_length choose a valid residue
        for sequence_idx, sequence in enumerate(self.initial_sequences):
            length_sequence = len(sequence)

            for residue_idx in range(length_sequence, self.max_length, 1):
                self.legal_actions_mask[sequence_idx, residue_idx] = 0

        # steps per episode
        steps_current_episode = 0


    def reset(self):

        self.state = {'Sequences': [sequence.__str__() for sequence in self.initial_sequences],
                      'Residue_positions': np.zeros( (self.nbr_sequences, self.max_length),dtype=np.int64 )
                      }

        for i, sequence in enumerate(self.state['Sequences']):
            # state is just the indices of each amino acide which is 1,2,3,4...len(sequence)+1 (taken from walids)
            self.state['Residue_positions'][i, :len(sequence)] = np.arange(1, len(sequence)+1)

        return self._get_obs()

    def _get_obs(self):
        return self.state

    # Possibly TODO (return sum of pair score or return entropy of columns, or frequency of collums)
    #def _get_info(self):

    def step(self, action):
        self.state = self.get_next_state(self.state, action)
        reward = self.get_value(self.state)
        self.steps_
        # set this to end after n steps
        done = False

        return self.state, reward, done

    def get_next_state(self, state, action):
        sequence, column = action
        new_state = {
            'Sequences': state['Sequences'].copy(),
            'Residue_positions': state['Residue_positions'].copy()
        }


        for i in range(column, len(state['Sequences'][sequence]),1):
            new_state['Residue_positions'][sequence][i] += 1

        return new_state


    def sum_pairs_score(self,state, scoringMatrix):
        max_length = max(max(sequence) for sequence in state['Residue_positions'])
        # TODO refactor
        score = 0
        # get number of blossum matrix
        match = re.search("blossum(.*)", scoringMatrix).group(1)
        score_matrix_dict = bl.BLOSUM(int(match))
        pairwise_combinations = itertools.combinations(np.arange(0, self.nbr_sequences), 2)

        for pair in pairwise_combinations:
            sequence_1, sequence_2 = pair
            for column in range(1, max_length+1, 1):

                residue_1 = '-'
                residue_2 = '-'

                array = np.where(state['Residue_positions'][sequence_1] == column )[0]
                if len(array) > 0:
                    index = array[0]
                    residue_1 = state['Sequences'][sequence_1][index]

                array = np.where( state['Residue_positions'][sequence_2] == column )[0]
                if len(array) > 0:
                    index = array[0]
                    residue_2 = state['Sequences'][sequence_2][index]

                if residue_1 == '-' or residue_2 == '-':
                    score -= 1
                else:
                    score += score_matrix_dict[residue_1 + residue_2]

        return score

    def get_value(self, state):
        return self.sum_pairs_score(state, 'blossum50')

    def get_alignment(self, state):
        sequences = state['Sequences'].copy()
        new_sequences = []

        max_len = np.max(state['Residue_positions'])
        # make a matrix full of gaps
        alignment = np.full([self.nbr_sequences, max_len], '-')

        for i, seq in enumerate(sequences):
            for j, base in enumerate(seq):
                alignment[i, state['Residue_positions'][i, j] - 1] = base
        return alignment
