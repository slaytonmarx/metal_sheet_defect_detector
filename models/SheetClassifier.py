import logging
import torch
import torch.nn as nn
from torch.nn import init

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SheetClassifier(nn.Module):
    '''A CNN-based classifier for sheet defect detection.'''
    def __init__(self, conv_size: int = 8, imsize: int = 32):
        super(SheetClassifier, self).__init__()
        try:
            self.model = nn.Sequential(
                nn.Conv2d(1, conv_size, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(conv_size),
                nn.Conv2d(conv_size, conv_size * 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(conv_size * 2)
            )

            # Initialize Convolutional Modules with Kaiming He's Method
            for m in self.model:
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, a=0.1)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            self.lin1 = nn.Linear(conv_size * 2 * imsize * imsize, 3)
            logging.info("SheetClassifier initialized successfully.")

        except Exception as e:
            logging.error(f"Error initializing SheetClassifier: {e}", exc_info=True)
    
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
            x = self.model(x)
            x = self.lin1(x.view(x.shape[0], -1))
            return x
        
        except Exception as e:
            logging.error(f"Error in forward pass: {e}", exc_info=True)
            return None
