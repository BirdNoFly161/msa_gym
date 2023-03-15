import gymnasium as gym
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

        sequences = list(map(self.sequence_constructor, sequences))
        self.grid = self.generate_grid(sequences)

    def generate_grid(self, sequences):

        max_length = max(len(sequence) for sequence in sequences)
        for idx, sequence in enumerate(sequences):
            if len(sequence) < max_length:
                sequences[idx] = self.sequence_constructor.concat([sequence, self.sequence_constructor('-' * (max_length-len(sequence)))])

        return sequences
