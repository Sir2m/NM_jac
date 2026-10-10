import torch
import numpy as np
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from preprocessing import BCICausalPreprocessor
from torch.utils.data import TensorDataset, DataLoader

def train_eval(subject, model, batch_size, criterion, optimizer, epochs, device, preprocessor):

    print("\nProcessing Training Data...")
    train_file = f'/kaggle/input/datasets/abdelrahmanyousryyu/bci-comp2a/A0{subject}T.gdf'
    train_eeg, train_events, train_dict = preprocessor.process_file(train_file, is_training=True)

    X_train, y_train, _ = BCICausalPreprocessor.generate_causal_windows(
        train_eeg, train_events, train_dict, window_size_sec=2.0, is_training=True
    )

    train_mean = np.mean(X_train, axis=-1, keepdims=True)
    train_std = np.std(X_train, axis=-1, keepdims=True) + 1e-8
    X_train_norm = (X_train - train_mean) / train_std

    train_loader = DataLoader(
            TensorDataset(torch.tensor(X_train_norm, dtype=torch.float32), 
                          torch.tensor(y_train, dtype=torch.long)),
            batch_size=batch_size, shuffle=True
        )

    model = model.to(device)
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
                
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
            logits = model(inputs)
        
            ce_loss = criterion(logits, labels)
            total_loss += ce_loss.item()
        
            ce_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        

        if (epoch + 1) % 10 == 0 or epoch == 0:
            train_acc = 100 * correct / total
            print(f"Epoch [{epoch+1}/{epochs}] | "
                f"CE Loss: {total_loss/len(train_loader):.4f} | "
                f"Train Acc: {train_acc:.2f}%")
            if (epoch+1)==epochs:
                print(f"\n{'='*50}")
                print(f"🏆 OFFICIAL FINAL TRAINING RESULTS (T)")
                print(f"{'='*50}")
                print(f"CE Loss: {total_loss/len(train_loader):.4f}")
                print(f"Train Acc: {train_acc:.2f}%")
                print(f"{'='*50}\n")

    print(f"Running Official Holdout Evaluation (Session T -> Session E) on device: {device}\n")
    print("Loading official evaluation true labels...")
    mat_path = f"/kaggle/input/datasets/abdelrahmanyousryyu/bci-competition-iv-data-sets-2a-true-labels/A0{subject}E.mat"

    mat_data = sio.loadmat(mat_path)

    # proccessing file data
    if 'classlabel' in mat_data:
        true_trial_labels = mat_data['classlabel'].flatten()
    elif 'labels' in mat_data:
        true_trial_labels = mat_data['labels'].flatten()
    else:
        for key, val in mat_data.items():
            if isinstance(val, np.ndarray) and val.flatten().shape[0] in [288, 43198]:
                true_trial_labels = val.flatten()
                break
                
    if true_trial_labels.min() == 1:
        true_trial_labels = true_trial_labels - 1

    print(f"Successfully loaded {len(true_trial_labels)} trial-level true keys.")



    print(f"\n=== Processing Subject {subject} Evaluation GDF ===")
    eval_file = f'/kaggle/input/datasets/abdelrahmanyousryyu/bci-comp2a/A0{subject}E.gdf'

    eval_eeg, eval_events, eval_dict = preprocessor.process_file(eval_file, is_training=False)
    X_eval, _, t_eval = BCICausalPreprocessor.generate_causal_windows(eval_eeg, eval_events, eval_dict, is_training=False)

    # Normalize Evaluation windows using TRAINING stats
    print("Applying global training normalization to evaluation data...")
    X_eval_norm = (X_eval - train_mean) / train_std

    print("Reconstructing trial indices from the relative timeline clock...")

    trial_mappings = np.zeros(len(t_eval), dtype=np.int64)
    current_trial_idx = 0

    for i in range(1, len(t_eval)):
        if t_eval[i] < t_eval[i-1] or t_eval[i] == 0.0:
            current_trial_idx += 1
        
        if current_trial_idx >= len(true_trial_labels):
            current_trial_idx = len(true_trial_labels) - 1
            
        trial_mappings[i] = current_trial_idx

    y_eval_true = true_trial_labels[trial_mappings]

    print(f"Labels successfully aligned! Verification Shape: {y_eval_true.shape}")
    print(f"Total unique trials reconstructed from timeline: {len(np.unique(trial_mappings))}/288")

    eval_loader = DataLoader(
        TensorDataset(torch.tensor(X_eval_norm, dtype=torch.float32), torch.tensor(y_eval_true, dtype=torch.long)),
        batch_size=64, shuffle=False
    )

    model.eval()

    # Dictionary to accumulate logits per trial
    trial_logits_accumulator = {i: [] for i in range(len(true_trial_labels))}

    with torch.no_grad():
        window_idx = 0 
        for inputs, labels in eval_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            
            logits_cpu = logits.cpu().numpy()
            
            for batch_i in range(logits.size(0)):
                current_trial = trial_mappings[window_idx]
                trial_logits_accumulator[current_trial].append(logits_cpu[batch_i])
                window_idx += 1

    # Aggregate and score
    eval_correct = 0
    all_true_labels = []
    all_pred_labels = []

    for trial_id, logit_list in trial_logits_accumulator.items():
        if len(logit_list) == 0:
            continue
                    
        # Sum/Mean all window logits within the trial
        mean_logits = np.mean(logit_list, axis=0)
        predicted_class = np.argmax(mean_logits)
        true_class = true_trial_labels[trial_id]
            
        # Store for the confusion matrix
        all_true_labels.append(true_class)
        all_pred_labels.append(predicted_class)
            
        if predicted_class == true_class:
            eval_correct += 1

    final_holdout_acc = 100 * eval_correct / len(true_trial_labels)

    print(f"\n{'='*50}")
    print(f"🏆 OFFICIAL TRIAL-LEVEL EVALUATION RESULTS (T -> E)")
    print(f"{'='*50}")
    print(f"Target Subject File:  A0{subject}E.gdf")
    print(f"Total True Trials:    {len(true_trial_labels)}")
    print(f"Final True Test Acc:  {final_holdout_acc:.2f}%")
    print(f"{'='*50}\n")

    class_names = ['Left Hand', 'Right Hand', 'Both Feet', 'Tongue']

    cm = confusion_matrix(all_true_labels, all_pred_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names,
                annot_kws={"size": 14})

    plt.title(f'Holdout Confusion Matrix (Acc: {final_holdout_acc:.1f}%)', fontsize=14, pad=15)
    plt.xlabel('Predicted MI Class', fontsize=12, labelpad=10)
    plt.ylabel('True MI Class', fontsize=12, labelpad=10)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()