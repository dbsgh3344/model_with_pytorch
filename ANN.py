import torch.nn as nn
import torch
import pandas as pd
import numpy as np


class ANN(nn.Module) :
    def __init__(self,input_dim,hidden_dim,output_dim) :
        super(ANN,self).__init__()

        self.fc1 = nn.Linear(input_dim,hidden_dim) 
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim,hidden_dim) 
        self.tanh2 = nn.Tanh()

        self.fc3 =nn.Linear(hidden_dim,hidden_dim)
        self.elu3 = nn.ELU()

        self.fc4 =nn.Linear(hidden_dim,output_dim)
        
        
    def forward(self,x) :
        
        out =self.fc1(x)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.tanh2(out)

        out = self.fc3(out)
        out=self.elu3(out)

        out = self.fc4(out)

        return out
