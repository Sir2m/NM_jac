import torch.nn as nn
import torch.optim as optim
from .loop import train_eval

def _run_(subject, architecture, parameters, batch_size, epochs, device, preprocessor):
    list = []
    for i in range(5):
        model = architecture(*parameters)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
        list.append(train_eval(subject, model, batch_size, criterion, optimizer, epochs, device, preprocessor))
        print(f"End of {i+1}")
        print("="*200)
    return list