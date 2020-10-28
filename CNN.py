import torch.nn as nn
import torch


class CNN (nn.Module) :
    def __init__(self) :
        super (CNN,self).__init__()
        
        self.cnn1 = nn.Conv2d(in_channels=1,out_channels=16, kernel_size=5, stride=1, )
        # output (batch,out_ch,H,W) = (100,16,24,24)
        self.relu1= nn.ReLU()

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        # output = (100,16,12,12)

        self.cnn2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1)
        # output =( 100,32, 8,8)
        self.relu2 =nn.ReLU()
        

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        #output =(100,32,4,4)
        

        self.fc1 = nn.Linear(32*4*4,10)
        

    def forward(self,x) :
            
        cnn1 =self.cnn1(x)
        relu1 = self.relu1(cnn1)

        mxp1 = self.maxpool1(relu1)
        
        cnn2 = self.cnn2(mxp1)
        relu2 = self.relu2(cnn2)

        mxp2 =self.maxpool2(relu2)
        mxp2 = mxp2.view(mxp2.size(0),-1)

        fc1 = self.fc1(mxp2)

        return fc1
