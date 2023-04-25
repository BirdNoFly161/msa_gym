class MultipleSequenceAlignmentEnv(gym.Env):
    def __init__(self, sequences):
        super(MultipleSequenceAlignmentEnv, self).__init__()
        self.score = 0
        self.state = None
        self.sequences = sequences
        self.n_sequences = len(sequences)
        self.max_length = max(len(sequence) for sequence in sequences)
        self.action_space = gym.spaces.MultiDiscrete([self.n_sequences, self.max_length])
        self.observation_space = gym.spaces.Dict({
            'sequences' : gym.spaces.Sequence(gym.spaces.Text(), maxlen=self.max_length),
            'positions' : gym.spaces.MultiDiscrete([self.n_sequences, self.max_length]),
            # profile could be a good idea for the model to know the distribution of amino acids in each position 
            #'profile' : gym.spaces.MultiBinary([self.max_length, self.n_characters])
        })
        
    def reset(self):
        self.state = np.zeros([self.n_sequences, self.max_length], dtype=np.int_) # watch out for overflow
        for i, sequence in enumerate(self.sequences):
            # state is just the indices of each amino acide which is 1,2,3,4...len(sequence)+1
            self.state[i, :len(sequence)] = np.arange(1, len(sequence)+1)
        self.score = self.column_score()
        return self._get_observation()
    
    def _get_observation(self):
        sequences = [''.join(list(sequence)) for sequence in self.state.astype(str)]
        return {
            'sequences': sequences,
            'positions': self.state,
        }
    
    def _insert_gap(self, seq_idx, pos):
        # inserting a gap in the sequence seq_idx at position pos
        self.state[seq_idx, pos:len(self.sequences[seq_idx])] += 1

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