import logging
import torch
import torch.nn as nn
import torchvision.models as models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ResnetClassifier(nn.Module):
    '''ResNet-based classifier with three output classes.'''
    def __init__(self):
        super(ResnetClassifier, self).__init__()
        try:
            self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            
            # Freeze all pre-trained layers
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Replace the last fully connected layer
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, 3)
            
            logging.info("ResnetClassifier initialized successfully.")

        except Exception as e:
            logging.error(f"Error initializing ResnetClassifier: {e}", exc_info=True)

    def get_accuracy(self, yhat, y):
        '''Computes the accuracy of a batch.'''
        try:
            correct_predictions = (torch.argmax(yhat, dim=1) == y).sum().item()
            accuracy = correct_predictions / y.size(0)
            return accuracy
        
        except Exception as e:
            logging.error(f"Error computing accuracy: {e}", exc_info=True)
            return 0.0

    def forward(self, x):
        '''Defines the forward pass.'''
        try:
            return self.model(x)
        
        except Exception as e:
            logging.error(f"Error in forward pass: {e}", exc_info=True)
            return None

