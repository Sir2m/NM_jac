import torch
import torch.nn as nn
from braindecode.models import EEGTCNet as BraindecoderEEGTCNet

"""
used parameters
(4, 22, 500, 0.5, 4, 4, 32, 2, 8, 12)
750 epoch
6960
56.9%
"""

class TCNet(nn.Module):    
    def __init__(
        self,
        nb_classes=4,
        Chans=22,
        Samples=500,
        dropoutRate=0.5,
        D=4,
        KT= 4,               # kernel size in TCN
        kernLength= 32,      # kernel size of first conv (phi^1)
        tcn_layers= 2,       # number of dilated residual blocks
        F1= 8,               # temporal filters
        FT= 12,              # number of filters in TCN
        ):
        super(TCNet, self).__init__()
        
        self.model = BraindecoderEEGTCNet(
            n_chans=Chans,
            n_outputs=nb_classes,
            n_times=Samples,
            drop_prob=dropoutRate,
            depth_multiplier=D,
            kernel_size=KT,
            kern_length=kernLength,
            depth=tcn_layers,
            filter_1=F1,
            filters=FT
        )
    
    def forward(self, x):
        """
        Forward pass through EEG-Conformer.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch, channels, time_points) or (batch, 1, channels, time_points)
        
        Returns:
        --------
        logits : torch.Tensor
            Classification logits of shape (batch, n_outputs)
        """
        # # Handle 3D input: (batch, channels, time) -> add channel dimension
        # if len(x.shape) == 3:
        #     x = x.unsqueeze(1)  # (batch, 1, channels, time)
        
        return self.model(x)
