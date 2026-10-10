import torch.nn as nn
import torch.optim as optim
from .NM_loop import NM_train_eval
from models import NM_PINN

def run_(subject, parameters, batch_size, epochs, device, preprocessor, lambda_ortho, max_lambda):
    list = []
    for i in range(5):
        model = NM_PINN(*parameters)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        physics_param_names = {'tau_E', 'tau_I', 'w_EE', 'w_EI', 'w_IE', 'w_II', 'P', 'Q'}
        physics_params = [p for n, p in model.named_parameters() if any(x in n for x in physics_param_names)]
        base_params = [p for n, p in model.named_parameters() if not any(x in n for x in physics_param_names)]
        
        optimizer = optim.AdamW([
            {'params': base_params, 'lr': 8e-4},
            {'params': physics_params, 'lr': 1e-5}
        ], weight_decay=1e-3)
        list.append(NM_train_eval(subject, model, batch_size, criterion, optimizer, epochs, device, preprocessor, lambda_ortho, max_lambda))
        print(f"End of {i+1}")
        print("="*200)
    return list
