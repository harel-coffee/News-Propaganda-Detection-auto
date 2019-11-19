''' IMPORTS '''

import torch
import torch.nn as nn

torch.manual_seed(1234)                                        #Seed set for replication


''' MODELS '''

class CharLSTM(nn.Module):
    '''
    Takes n-characters from both side and concatenates the output to a sigmoid layer
    '''
    
    def __init__(self, input_size, hidden_size, dropout):
        super(CharLSTM, self).__init__()
        
        #Input Parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        #Defining Layers
        self.lstm_l = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
        self.lstm_r = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout)
        self.fc = nn.Linear(2*hidden_size, 1)
        
    def forward(self, sequence_r, sequence_l):
        #LSTM Outputs
        lstm_r_out, _ = self.lstm_r(sequence_r)                 #Right LSTM outputs
        lstm_l_out, _ = self.lstm_l(sequence_l)                 #Left LSTM outputs
        
        #FC Layer
        fc_input = torch.cat(lstm_r_out[-1], lstm_l_out[-1])    #Concatenating outputs from last cells of both LSTMs
        out = nn.Sigmoid(fc(fc_input))                          #Final sigmoid output
        
        return out
        
        