import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.parametrize import register_parametrization

import mne
import braindecode
import numpy as np
import seaborn as sns
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from sklearn.metrics import confusion_matrix

from preprocessing import BCICausalPreprocessor

from models import EEG_Conformer, EEG_Net, TCNet, NM_PINN
from train import _run_, run_

# parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Initializing Preprocessor...")
preprocessor = BCICausalPreprocessor(lowcut=8.0, highcut=30.0)
batch_size = 64
epochs = 300
lists = []
parameters = (22, 8, 2, 4, 250, 16)

lists.append(run_(1, parameters, batch_size, epochs, device, preprocessor, 0.01, 0.001))
lists.append(run_(2, parameters, batch_size, epochs, device, preprocessor, 0.01, 0.001))
lists.append(run_(3, parameters, batch_size, epochs, device, preprocessor, 0.01, 0.001))
lists.append(run_(4, parameters, batch_size, epochs, device, preprocessor, 0.01, 0.001))
lists.append(run_(5, parameters, batch_size, epochs, device, preprocessor, 0.01, 0.001))
lists.append(run_(6, parameters, batch_size, epochs, device, preprocessor, 0.01, 0.001))
lists.append(run_(7, parameters, batch_size, epochs, device, preprocessor, 0.01, 0.001))
lists.append(run_(8, parameters, batch_size, epochs, device, preprocessor, 0.01, 0.001))
lists.append(run_(9, parameters, batch_size, epochs, device, preprocessor, 0.01, 0.001))

print(lists)

maxes = [max(row) for row in lists]
print(maxes)
print(np.mean(maxes))
model = NM_PINN(*parameters)
print(summary(model))