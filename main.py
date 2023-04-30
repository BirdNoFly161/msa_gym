from MsaGrid import MsaGrid
import torch
from Model import ResNet
from MCTS import MCTS
import numpy as np
from torchsummary import summary
from AlphaZero import AlphaZero
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    msaGrid = MsaGrid(sequences_filename='test2.fa', sequence_constructor='protein')
    msaGrids = []

    files_name_list= ['12.fa',
    '12s112.fa',
    '12s117.fa',
    '12t113.fa',
    '12t116.fa',
    '12t117.fa',
    '12t119.fa',
    '22.fa',
    '22t53.fa',
    '22t56.fa']
    for filename in files_name_list:
        msaGrids.append(MsaGrid(sequences_filename=filename, sequence_constructor='protein'))

    state = msaGrid.reset()

    print('Score at start: ', msaGrid.get_value(state))
    print('Alignment at start: ', msaGrid.get_alignment(state))

    model = ResNet(msaGrid, num_resBlocks=3, num_hidden=64, device= device)
    #model.load_state_dict(torch.load('model_90.pt'))
    model.eval()
    summary(model, (4, 6, 30))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer.load_state_dict(torch.load('optimizer_90.pt'))

    args = {
        'C': 20,
        'num_searches': 1000 ,
        'num_iterations': 10,
        'num_selfPlay_iterations': 5,
        'num_epochs': 2,
        'batch_size': 3
    }

    #alphaZero = AlphaZero(model, optimizer, msaGrid, args)
    #alphaZero.learn()

    """
    memory = alphaZero.selfPlay()

    for state in memory:
        print(state[0])
        print(msaGrid.get_alignment(state[0]))
        print(msaGrid.get_value(state[0]))

    """
    for grid in msaGrids:
        alphaZero = AlphaZero(model, optimizer,grid , args)
        #alphaZero.learn()
        memory = alphaZero.selfPlay()
        for state in memory:
            print(state[0])
            print(msaGrid.get_alignment(state[0]))
            print(msaGrid.get_value(state[0]))


