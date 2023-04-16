from MsaGrid import MsaGrid
from MCTS import MCTS
import numpy as np
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    msaGrid = MsaGrid(sequences_filename='12s112.fa', sequence_constructor='protein')

    state = msaGrid.reset()

    """
    #print(msaGrid.find_nth_occurrence(msaGrid.grid[0],'-',6))
    print(msaGrid.slide_grid(msaGrid.grid, sequence= 0, residue_position= 1, distance= -2))
    #print(msaGrid.get_valid_moves(msaGrid.grid))
    print(msaGrid.grid)
    print(msaGrid.get_value(msaGrid.grid))
    
    """

    #print(msaGrid.get_valid_moves(msaGrid.grid)[0])
    args = {
        'C': 1.41,
        'num_searches': 1000
    }

    mcts = MCTS(msaGrid, args)
    #"""
    while True:
        print(state)
        print('Score: ',msaGrid.get_value(state))

        mcts_probs = mcts.search(state)
        action = np.unravel_index(np.argmax(mcts_probs), mcts.MSA.action_size)
        sequence, column = action
        distance = 0
        if (column % 2 == 1):
            distance = 1
        else:
            distance = -1
        column = int(column / 2)
        state, legally_achieved = mcts.MSA.get_next_state(state, sequence, column, distance)

    #"""

