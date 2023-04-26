from MsaGrid import MsaGrid
import torch
from Model import ResNet
from MCTS import MCTS
import numpy as np
from torchsummary import summary

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    msaGrid = MsaGrid(sequences_filename='test2.fa', sequence_constructor='protein')

    state = msaGrid.reset()

    print('Score at start: ', msaGrid.get_value(state))
    print('Alignment at start: ', msaGrid.get_alignment(state))

    model = ResNet(msaGrid, 1, 64)
    summary(model, (4, 6, 29))

    args = {
        'C': 20,
        'num_searches': 1000
    }
    
    model = ResNet(msaGrid, 1, 64)
    
    mcts = MCTS(msaGrid, args, model)

    steps = 100
    
    while steps > 0:
        print(state)
        print('Score: ',msaGrid.get_value(state))
        print('Alignment: \n', msaGrid.get_alignment(state))

        mcts_probs = mcts.search(state)
        action = np.unravel_index(np.argmax(mcts_probs), (msaGrid.nbr_sequences, msaGrid.max_length))

        state= mcts.MSA.get_next_state(state, action)
        steps -=1



    ## delete below later once i make sure i dont need them
    print('maximum length:' ,msaGrid.max_length)
    print('alphabet size:' ,len(msaGrid.sequence_constructor.alphabet))
    print(msaGrid.one_hot_encode(state))

    encoded_state = msaGrid.get_encoded_state(state)

    tensor_state = torch.tensor(encoded_state).unsqueeze(0)

    model = ResNet(msaGrid, 1, 64)
    summary(model, (4, 6, 29))

    policy, value = model(tensor_state)
    value = value.item()
    policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

    print(value, policy)
