from torch.nn import init
import torch.nn as nn
import torch

class SheetClassifier(nn.Module):
    def __init__(self, conv_size:int = 8, imsize:int = 32):
        super(SheetClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, conv_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(conv_size),
            nn.Conv2d(conv_size, conv_size*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(conv_size*2)
            #nn.MaxPool2d(3),
        )

        # Initialize Convolutional Modules with Kaiming He's Method
        # for m in self.model:
        #     if m.__class__ == nn.Conv2d: init.kaiming_normal_(m.weight, a=0.1)

        self.lin1 = nn.Linear(conv_size*2*imsize*imsize, 3)
    
    def get_accuracy(self, yhat, y):
        '''Returns the accuracy of the batch'''
        return 1 - abs(torch.argmax(yhat,dim=1) - y).clip(0,1).sum()/len(yhat)

    def forward(self, x):
        x = self.model(x)
        x = self.lin1(x.view(x.shape[0],-1))
        return x