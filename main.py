from MsaGrid import MsaGrid
from MCTS import MCTS
import numpy as np
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    msaGrid = MsaGrid(sequences_filename='12.fa', sequence_constructor='protein')

    state = msaGrid.reset()

    print('Score at start: ', msaGrid.get_value(state))
    print('Alignment at start: ', msaGrid.get_alignment(state))

    args = {
        'C': 20,
        'num_searches': 500
    }


    mcts = MCTS(msaGrid, args)

    steps = 10

    
    while steps > 0:
        print(state)
        print('Score: ',msaGrid.get_value(state))
        print('Alignment: \n', msaGrid.get_alignment(state))

        mcts_probs = mcts.search(state)
        action = np.unravel_index(np.argmax(mcts_probs), (msaGrid.nbr_sequences, msaGrid.max_length))

        state= mcts.MSA.get_next_state(state, action)
        steps -=1

