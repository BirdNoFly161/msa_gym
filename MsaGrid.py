import gymnasium as gym
import skbio.io
from skbio import Protein, DNA


class MsaGrid(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 5}

    def __init__(self, render_mode=None, sequences_filename=None, sequence_constructor=None):
        constructors = {'protein': Protein, 'dna': DNA}
        self.sequence_constructor = constructors[sequence_constructor]

        # why put them into Protein or DNA objects the RL agent will only see string representations?
        self.sequences = list(skbio.read(sequences_filename, format='fasta'))
        self.sequences = list(map(self.sequence_constructor, self.sequences))
        self.grid = self.generate_grid(self.sequences)
        # MSA object returned by skbio should has a shape attribute
        self.n_sequences = len(self.sequences)
        # but i don't know for sure
        self.max_length = max(len(sequence) for sequence in self.sequences)
        # gym settings
        # not sure how this will change with stepping
        self.observation_shape = (self.n_sequences, self.max_length)
        self.action_shape = (self.n_sequences, self.max_length)
        self.action_space = gym.spaces.MultiDiscrete(
            self.action_shape)  # not sure if this is the right space
        # side note: maybe use the sequences and the column wise profiles as the observation space?
        # this would be achived  gym.spaces.Dict(seuqencs,profiles)

    def generate_grid(self, sequences):
        # should be a n*max_length array?
        # this will be very expensive as with each inseration the matrix will have to expand max_length++
        for idx, sequence in enumerate(sequences):
            # if len(sequence) < max_length:
            padding_length = self.max_length - len(sequence)
            padding_sequence = self.sequence_constructor('-' * padding_length)
            sequences[idx] = self.sequence_constructor.concat([sequence, padding_sequence])
        return sequences
    def reset(self):
        # reset the grid to the original state
        # self.grid = self.generate_grid(self.sequences)
        # return self.grid
        raise NotImplementedError
    def render(self, mode='human'):
        raise NotImplementedError
