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

        self.action_size = None


    def reset(self):
        padded_sequences = list()
        for idx, sequence in enumerate(self.initial_sequences):
            if len(sequence) < self.max_length:
                padded_sequences.append( self.sequence_constructor.concat(
                    [sequence, self.sequence_constructor('-' * (self.max_length - len(sequence)))])
                )
        self.grid = [sequence.__str__() for sequence in padded_sequences]

        # for testing purposes only
        self.grid = ['FGKGKC-',
       'FGKFGK-',
       '-GKGKC-',
       'KFKC---']

        # move this line to init after test
        shape_grid = (len(self.grid), len(self.grid[0]))
        self.action_size = (shape_grid[0], shape_grid[1] * 2)
        return self._get_obs()

    def _get_obs(self):
        return tuple(self.grid)

    # Possibly TODO (return sum of pair score or return entropy of columns, or frequency of collums)
    #def _get_info(self):

    def generate_grid(self,sequences):
        ...

    """
        print(self.sequence_constructor.alphabet)
        print(self.sequence_constructor.gap_chars)
        self.observation_space = spaces.Sequence( spaces.Text(charset= self.sequence_constructor.alphabet, max_length=max_length) )
        print(self.observation_space.sample())
        print(type(self.observation_space.sample()))
        return [sequence]
    """
    def find_nth_occurrence(self, string, substring, n):
        start = string.find(substring)
        while start >= 0 and n > 1:
            start = string.find(substring, start + len(substring))
            n -= 1
        return start

    def slide_grid(self, grid, sequence, residue_position, distance):

        grid = list(grid).copy()

        if distance > 0:
            direction = 'right'
        else:
            distance *= -1
            direction = 'left'

        if direction == 'right':

            new_sequence = {'left': grid[sequence][:residue_position],
                            'right': grid[sequence][residue_position:]}

            idx_last_gap = self.find_nth_occurrence(new_sequence['right'], '-', distance)
            if idx_last_gap > 0:
                #print(''.join(['-'*distance, 'hello']))
                new_part = ''.join([ ('-' * distance), new_sequence['right'][:idx_last_gap].replace('-', ''), new_sequence['right'][idx_last_gap+1:]])
                grid[sequence] = ''.join([new_sequence['left'], new_part])
                #return(''.join([new_sequence['left'], new_part]))

            else:
                print('illegal attempt for state: ', grid, sequence, residue_position, distance)

            return grid, True

        else:
            new_sequence = {'left': grid[sequence][:residue_position+1],
                            'right': grid[sequence][residue_position+1:]}

            new_sequence['left'] = new_sequence['left'][::-1]
            #print(new_sequence['left'])
            idx_last_gap = self.find_nth_occurrence(new_sequence['left'], '-', distance)

            if idx_last_gap > 0:
                new_part = ''.join([ ('-' * distance), new_sequence['left'][:idx_last_gap].replace('-', ''), new_sequence['left'][idx_last_gap+1:]])

                grid[sequence] = ''.join([new_part[::-1], new_sequence['right'] ])
                #print(grid)
                #return(''.join([new_part[::-1], new_sequence['right'] ]))


            else:
                print('illegal attempt for state: ', grid, sequence, residue_position, distance)

            return grid, True




    def get_valid_moves(self, state):
        shape_state = (len(state), len(state[0]))
        valid_moves_mask = np.zeros( (shape_state[0], shape_state[1]*2) )


        for (i, j), element in np.ndenumerate(valid_moves_mask):
            # element here is redundant dont need its value just the indices

            residue_idx = int(j / 2)
            if (state[i][residue_idx] != '-'):
                new_sequence = {'left': state[i][:residue_idx],
                        'right': state[i][residue_idx:]}

                if new_sequence['left'].find('-')!=-1:
                    valid_moves_mask[i][residue_idx*2]=1

                if new_sequence['right'].find('-')!=-1:
                    valid_moves_mask[i][(residue_idx*2)+1]=1


        return valid_moves_mask

    #wrapper for my slide grid fuction, to conform to existing projects
    def get_next_state(self, state, sequence, column, distance):
        return self.slide_grid(state, sequence, column, distance)

    def compute_score_pairwise_alignment(self, seq1, seq2, score_matrix_dict):
        score = 0
        for residue_pair in list(zip(seq1, seq2)):
            residue1 = residue_pair[0].__str__()
            residue2 = residue_pair[1].__str__()

            # blosum matrices in the blosum modules read gaps as '*' instead of '-'
            if (residue1 == '-'):
                residue1 = '*'

            if (residue2 == '-'):
                residue2 = '*'

            if (residue1 == '*' and residue2 == '*'):
                score -= 1
            else:
                # add cost of substuting residue 1 with 2
                score += score_matrix_dict[residue1 + residue2]
        return score

    def sum_pairs_score(self,msa, scoringMatrix):
        score = 0
        # get number of blossum matrix
        match = re.search("blossum(.*)", scoringMatrix).group(1)
        score_matrix_dict = bl.BLOSUM(int(match))
        pairwise_combinations = itertools.combinations(msa, 2)

        for pair in pairwise_combinations:
            score += self.compute_score_pairwise_alignment(pair[0], pair[1], score_matrix_dict)

        return score

    def get_value(self, state):
        return self.sum_pairs_score(state, 'blossum50')