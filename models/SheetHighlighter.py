import torch.nn as nn
import numpy as np

class SheetHighlighter(nn.Module):
    def __init__(self):
        super(SheetHighlighter, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
            #nn.BatchNorm2d(1),
            #nn.AdaptiveAvgPool2d(output_size=(64,64))
            #nn.MaxPool2d(3),
        )
    
    def get_accuracy(self, yhat, y):
        '''Returns the accuracy of the batch'''
        return 1-(yhat - y).clip(0,1).sum()/yhat.numel()

    def forward(self, x):
        x = self.model(x)
        return x