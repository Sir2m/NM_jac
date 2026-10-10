import torch
import torch.nn as nn
from braindecode.models import EEGNet as BraindecoderEEGNet


"""
used parameters
(4, 22, 500, 8, 2, 32, 4, 8, 22, 0.5, 0.25) EEGNet8,2
800 epoch
3332
64.08%
"""

class EEG_Net(nn.Module):
    """
    Original paper:
        Lawhern et al., "EEGNet: A Compact Convolutional Neural Network for
        EEG-based Brain-Computer Interfaces", J. Neural Eng. 2018.

    Args:
        nb_classes  : number of output classes
        Chans       : number of EEG channels (electrodes)
        Samples     : number of time samples per trial
        dropoutRate : dropout probability
        kernLength  : length of the temporal convolution kernel (Block 1)
        F1          : number of temporal filters
        D           : depth multiplier for the depthwise conv
        F2          : number of pointwise filters (= F1 * D by convention)
        norm_rate   : max-norm constraint on the final Dense layer
        dropoutType : 'Dropout' or 'SpatialDropout2D'
    """
    
    def __init__(
        self,
        nb_classes,
        Chans=64,
        Samples=128,
        F1=8,
        D=2,
        F2=16,
        pk1=4,
        pk2=8,
        kernLength=64,
        dropoutRate=0.5,
        norm_rate=0.25,
    ):
        super(EEG_Net, self).__init__()
        
        self.model = BraindecoderEEGNet(
            n_chans=Chans,
            n_outputs=nb_classes,
            n_times=Samples,
            drop_prob=dropoutRate,
            pool1_kernel_size=pk1,
            pool2_kernel_size=pk2,
            F1=F1,
            D=D,
            F2=F2,
            kernel_length=kernLength,
            norm_rate=norm_rate
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
        
        return self.model(x)



