import os
import numpy as np

import torch
from torch.optim import Adam
from torchvision import models
from fastshap import ImageSurrogate
from fastshap.utils import MaskLayer2d, KLDivLoss, DatasetInputOnly

class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')


    def visualise_layer(self, image, IS_SURR = False):
        
        x = image
        if IS_SURR:

        for index, layer in enumerate(self.model):
            # Forward pass layer by layer
            x = layer(x)
            if index == self.selected_layer:
                # Only need to forward until the selected layer is reached
                # Now, x is the output of the selected layer
                break
        
        self.conv_output = x[0, self.selected_filter]