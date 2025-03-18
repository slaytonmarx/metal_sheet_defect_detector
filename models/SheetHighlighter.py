import logging
import torch
import torch.nn as nn

class SheetHighlighter(nn.Module):
    '''A CNN-based model to highlight sheet defects.'''
    def __init__(self):
        super(SheetHighlighter, self).__init__()
        try:
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
            )
            logging.info("SheetHighlighter initialized successfully.")

        except Exception as e:
            logging.error(f"Error initializing SheetHighlighter: {e}", exc_info=True)
    
    def get_accuracy(self, yhat, y):
        '''Computes the accuracy of the batch.'''
        try:
            accuracy = 1 - (yhat - y).clip(0, 1).sum() / yhat.numel()
            return accuracy.item()
        
        except Exception as e:
            logging.error(f"Error computing accuracy: {e}", exc_info=True)
            return 0.0
    
    def forward(self, x):
        '''Defines the forward pass.'''
        try:
            x = self.model(x)
            return x
        
        except Exception as e:
            logging.error(f"Error in forward pass: {e}", exc_info=True)
            return None
