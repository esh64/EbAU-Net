import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np

from data import DriveDataset
from loss import DiceBCELoss
from utils import seeding, create_dir, epoch_time
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate_with_metrics(model, loader, loss_fn, device):
    epoch_loss = 0.0
    metrics = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            batch_metrics = calculate_metrics(y, y_pred)
            metrics.append(batch_metrics)

    avg_metrics = np.mean(metrics, axis=0)
    return epoch_loss / len(loader), {
        "jaccard": avg_metrics[0],
        "f1": avg_metrics[1],
        "recall": avg_metrics[2],
        "precision": avg_metrics[3],
        "accuracy": avg_metrics[4],
    }


if __name__ == "__main__":
    """ Lista de modelos """
    model_files = [
       "modelOriginal.py",
       "modelSCSEAllConv2.py",
       "modelSCSE_Encoder_Decoder.py",
       "modelSCSE_Skip.py",
       "modelSCSE_Skip_Encoder_Decoder.py",
       "modelCSEAllConv2.py",
       "modelCSE_Encoder_Decoder.py",
       "modelCSE_Skip.py",
       "modelCSE_Skip_Encoder_Decoder.py",
       "modelSSEAllConv2.py",
       "modelSSE_Encoder_Decoder.py",
       "modelSSE_Skip.py",
       "modelSSE_Skip_Encoder_Decoder.py",
    ]

    """ Seeding e diretórios """
    seeding(42)
    create_dir("files")

    """ Load dataset """
    all_x = sorted(glob("data/train/image/*"))
    all_y = sorted(glob("data/train/mask/*"))

    print(f"Total Dataset Size: {len(all_x)}")

    """ Hyperparameters """
    H, W = 256, 256
    batch_size = 2
    num_epochs = 100
    lr = 1e-4

    """ Device """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    """ Treinamento de modelos """
    for model_file in model_files:
        module_name = model_file.replace(".py", "")  # Remove a extensão ".py"
        model_name = module_name.replace("model", "")  # Extrai o nome do modelo após "model"

        # Importar dinamicamente o módulo e a função `build_unet`
        imported_module = __import__(module_name)
        build_unet = imported_module.build_unet

        for seed in range(1, 6):
            print(f"\nTraining {model_name}, mdeol {seed}/5 with {seed}" seed)

            """ Configurar seeds """
            torch.manual_seed(seed)
            np.random.seed(seed)

            """ Split dataset """
            train_x, valid_x, train_y, valid_y = train_test_split(all_x, all_y, test_size=0.1, random_state=seed)

            """ Dataset e loader """
            train_dataset = DriveDataset(train_x, train_y)
            valid_dataset = DriveDataset(valid_x, valid_y)

            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4
            )

            valid_loader = DataLoader(
                dataset=valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )

            """ Modelo, otimizador, scheduler, loss """
            model = build_unet()
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
            loss_fn = DiceBCELoss()

            """ Treinamento """
            best_valid_loss = float("inf")
            checkpoint_path = f"files/{model_name}_{seed}.h5"

            for epoch in range(num_epochs):
                start_time = time.time()

                """ Treinamento """
                train_loss = train(model, train_loader, optimizer, loss_fn, device)

                """ Validação com métricas """
                valid_loss, valid_metrics = evaluate_with_metrics(model, valid_loader, loss_fn, device)

                """ Salvar modelo se melhorar """
                if valid_loss < best_valid_loss:
                    print(f"Val. loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving {checkpoint_path}")
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), checkpoint_path)

                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                print(
                    f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s\n"
                    f"\tTrain Loss: {train_loss:.3f}\n"
                    f"\t Val. Loss: {valid_loss:.3f}\n"
                    f"\t Val. Metrics: Jaccard: {valid_metrics['jaccard']:.4f}, "
                    f"F1: {valid_metrics['f1']:.4f}, Recall: {valid_metrics['recall']:.4f}, "
                    f"Precision: {valid_metrics['precision']:.4f}, Accuracy: {valid_metrics['accuracy']:.4f}\n"
                )

