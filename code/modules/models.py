''' IMPORTS '''

import torch
import torch.nn as nn

torch.manual_seed(1234)                                        #Seed set for replication


''' MODELS '''

class WordTDLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size):
        super(WordTDLSTM, self).__init__()
        
        #Input Parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        #Defining Layers
        self.lstm_l = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.lstm_r = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(2*hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, sequence_r, sequence_l):
        #LSTM Outputs
        __, (lstm_r_out, _) = self.lstm_r(sequence_r)                 #Right LSTM output
        __, (lstm_l_out, _) = self.lstm_l(sequence_l)                 #Left LSTM output

        #FC Layer
        fc_input = torch.cat((lstm_r_out, lstm_l_out), dim=2)    #Concatenating outputs from last cells of both LSTMs
        fc_input = fc_input.reshape(fc_input.shape[1], fc_input.shape[2])
        out = self.fc(fc_input)                                       #Final sigmoid output
        out = self.sigmoid(out)
        
        return out
        
        