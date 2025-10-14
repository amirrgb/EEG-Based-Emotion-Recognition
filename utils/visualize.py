import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import os
import datetime

global plots_save_dir
plots_save_dir = "../results/plots"

def plot_loss_and_acc_curve(train_accs, test_accs, train_losses, test_losses,model_name):
    os.makedirs(plots_save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    acc_path = os.path.join(plots_save_dir, f"{model_name}_accuracy_{timestamp}.png")
    loss_path = os.path.join(plots_save_dir, f"{model_name}_loss_{timestamp}.png")

    plt.figure(figsize=(10, 4))
    plt.plot(train_accs, label='Train Accuracy', marker='o')
    plt.plot(test_accs, label='Test Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'{model_name} Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.show()

    # Loss Plot
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(test_losses, label='Test Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.show()

    return acc_path, loss_path

def get_all_preds_and_true(model, test_loader, device, num_classes, change_train_func=False,sigmoid_threshold=0.5):
    all_preds = []
    all_true = []

    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            if num_classes == 2:
                y = y.float().unsqueeze(1)
            out = model(X)[0] if change_train_func else model(X)
            if num_classes == 2:
                preds = torch.sigmoid(out)
                preds = (preds > sigmoid_threshold).int().squeeze()
            else:
                preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(y.cpu().numpy())

    return all_preds, all_true

def plot_confusion_matrix(all_preds, all_trues,num_classes,model_name):
    os.makedirs(plots_save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cm_path = os.path.join(plots_save_dir, f"{model_name}_confusion_matrix_{timestamp}.png")

    cm = confusion_matrix(all_trues, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f"Emo {i}" for i in range(num_classes)],
                yticklabels=[f"Emo {i}" for i in range(num_classes)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.show()

    return cm_path

def plot_classification_report(all_preds, all_trues,model_name):
    os.makedirs(plots_save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(plots_save_dir, f"{model_name}_classification_report_{timestamp}.png")

    report = classification_report(all_trues, all_preds, output_dict=True, digits=4)
    print(report)
    df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.iloc[:-1, :].T, annot=True, cmap='Blues')
    plt.title('Classification Report')
    plt.xlabel('Classes')
    plt.ylabel('Metrics')
    plt.tight_layout()
    plt.savefig(report_path)
    plt.show()

    return report_path

def plot_attention_weights(attn_weights):
    plt.figure(figsize=(10, 6))
    sns.heatmap(attn_weights, cmap='Blues', cbar=True)
    plt.title('Attention Weights')
    plt.xlabel('Time Steps')
    plt.ylabel('Samples')
    plt.tight_layout()
    plt.show()

    # previous code :
    # X_sample, y_sample = test_dataset[0]  # first EEG sample
    # X_sample = X_sample.unsqueeze(0).to(device)  # add batch dim

    # model.eval()
    # with torch.no_grad():
    #     _, attn_weights = model(X_sample)

    # attn_weights = attn_weights.cpu().numpy()

    # plt.figure(figsize=(12, 4))
    # plt.plot(attn_weights)
    # plt.title("Attention Weights over Time (sample 0)")
    # plt.xlabel("Time step")
    # plt.ylabel("Attention weight")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

def plot_attention_on_eeg_signal():
    pass
    #previous code : 
    # channel_name = 'Cz'
    # channel_names = [
    #     "Fp1","Fp2","Fz","F3","A2","CP1","CP2","CP5",
    #     "C4","C3","T4","T3","O1","Oz","FC1","FC2",
    #     "PO3","PO4","T5","T6","Cz","P4","FC5","FC6",
    #     "O2","F8","F7","F4","A1","P3","Pz","CP6"
    # ]
    # channel_idx = channel_names.index(channel_name)

    # X_sample, y_sample = test_dataset[0]
    # signal = X_sample[:, channel_idx].numpy()  # (T,)
    # X_sample = X_sample.unsqueeze(0).to(device)

    # model.eval()
    # with torch.no_grad():
    #     _, attn_weights = model(X_sample)
    # attn_weights = attn_weights.cpu().numpy()

    # # normalize attention to [0, 1] for plotting
    # attn_norm = (attn_weights - attn_weights.min()) / (attn_weights.max() - attn_weights.min())

    # fig, ax1 = plt.subplots(figsize=(14, 4))

    # # EEG signal
    # ax1.plot(signal, color='blue', label=f'EEG signal ({channel_name})')
    # ax1.set_ylabel('EEG Amplitude', color='blue')
    # ax1.tick_params(axis='y', labelcolor='blue')

    # # Attention (scaled)
    # ax2 = ax1.twinx()
    # ax2.plot(attn_norm, color='red', alpha=0.6, label='Attention weight')
    # ax2.set_ylabel('Attention Weight (scaled)', color='red')
    # ax2.tick_params(axis='y', labelcolor='red')

    # plt.title(f"Attention + EEG Signal for channel {channel_name}")
    # fig.tight_layout()
    # plt.show()


