from MsaGrid import MsaGrid
import torch
from Model import ResNet
from MCTS import MCTS
import numpy as np
from torchsummary import summary
from AlphaZero import AlphaZero

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    msaGrid = MsaGrid(sequences_filename='test2.fa', sequence_constructor='protein')

    state = msaGrid.reset()

    print('Score at start: ', msaGrid.get_value(state))
    print('Alignment at start: ', msaGrid.get_alignment(state))

    model = ResNet(msaGrid, 1, 64)
    summary(model, (4, 6, 30))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    args = {
        'C': 2,
        'num_searches': 50,
        'num_iterations': 5,
        'num_selfPlay_iterations': 100,
        'num_epochs': 2,
        'batch_size': 30
    }

    alphaZero = AlphaZero(model, optimizer, msaGrid, args)
    alphaZero.learn()

    memory = alphaZero.selfPlay()
    print(memory[-1])


    """
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
        
        
    """

