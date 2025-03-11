import torchvision.models as models
import torch.nn as nn
import torch

class ResnetClassifier(nn.Module):
    def __init__(self):
        super(ResnetClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters(): param.requires_grad=False

        # Replace the last layer of the net with our own (if we wanted to replace it with multiple layers we could use nn.Seqeuntial)
        self.model.fc = nn.Linear(512, 3)
    
    def get_accuracy(self, yhat, y):
        '''Returns the accuracy of the batch'''
        return 1 - abs(torch.argmax(yhat,dim=1) - y).clip(0,1).sum()/len(yhat)

    def forward(self, x):
        x = self.model(x)
        #x = self.lin1(x.view(x.shape[0],-1))
        return x