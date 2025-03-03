import torch.nn as nn
import torch

class SheetClassifier(nn.Module):
    def __init__(self, imsize:int = 64):
        super(SheetClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16)
            #nn.MaxPool2d(3),
        )
        self.lin1 = nn.Linear(16*64*64, 3)
    
    def get_accuracy(self, yhat, y):
        '''Returns the accuracy of the batch'''
        return 1 - abs(torch.argmax(yhat,dim=1) - y).clip(0,1).sum()/len(yhat)

    def forward(self, x):
        x = self.model(x)
        x = self.lin1(x.view(x.shape[0],-1))
        return x