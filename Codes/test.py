import os
import time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score
import imageio  # Certifique-se de que esta biblioteca está instalada

from utils import create_dir, seeding


def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true_flat = y_true.reshape(-1)  # Flatten for AUC
    y_true_binary = (y_true > 0.5).astype(np.uint8).reshape(-1)  # Binary for discrete metrics

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred_flat = y_pred.reshape(-1)  # Flatten for AUC
    y_pred_binary = (y_pred > 0.5).astype(np.uint8).reshape(-1)  # Binary for discrete metrics

    """ Metrics """
    score_jaccard = jaccard_score(y_true_binary, y_pred_binary)
    score_f1 = f1_score(y_true_binary, y_pred_binary)
    score_recall = recall_score(y_true_binary, y_pred_binary)
    score_precision = precision_score(y_true_binary, y_pred_binary)
    score_acc = accuracy_score(y_true_binary, y_pred_binary)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)  ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask


if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("results")

    """ Load dataset """
    test_x = sorted(glob("data/test/image/*"))
    test_y = sorted(glob("data/test/mask/*"))

    """ Hyperparameters """
    H = 256
    W = 256
    size = (W, H)

    """ Models to test """
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

    """ Device """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for model_file in model_files:
        module_name = model_file.replace(".py", "")  # Remove extensão .py
        model_name = module_name.replace("model", "")  # Nome do modelo sem prefixo "model"
        print(f"Testing {model_name}")

        """ Import dynamic module """
        imported_module = __import__(module_name)
        build_unet = imported_module.build_unet

        """ Paths for all seeds """
        model_paths = [f"files/{model_name}_{i}.h5" for i in range(1, 6)]

        all_metrics = []

        for model_idx, model_path in enumerate(model_paths):
            print(f"  Processing Seed {model_idx + 1}")

            """ Load the model """
            model = build_unet()
            model = model.to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Added AUC
            time_taken = []

            for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
                """ Reading image """
                image = cv2.imread(x, cv2.IMREAD_COLOR)  ## (512, 512, 3)
                x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
                x = x / 255.0
                x = np.expand_dims(x, axis=0)          ## (1, 3, 512, 512)
                x = torch.from_numpy(x).to(device, dtype=torch.float32)

                """ Reading mask """
                mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
                y = np.expand_dims(mask, axis=0)           ## (1, 512, 512)
                y = y / 255.0
                y = np.expand_dims(y, axis=0)             ## (1, 1, 512, 512)
                y = torch.from_numpy(y).to(device, dtype=torch.float32)

                with torch.no_grad():
                    """ Prediction and FPS calculation """
                    start_time = time.time()
                    pred_y = model(x)
                    pred_y = torch.sigmoid(pred_y)
                    time_taken.append(time.time() - start_time)

                    score = calculate_metrics(y, pred_y)
                    metrics_score = list(map(add, metrics_score, score))

            """ Average metrics for the current seed """
            avg_metrics = [m / len(test_x) for m in metrics_score]
            fps = 1 / np.mean(time_taken)
            avg_metrics.append(fps)  # Add FPS as a metric
            all_metrics.append(avg_metrics)

        """ Save metrics to CSV """
        columns = ["Jaccard", "F1", "Recall", "Precision", "Accuracy", "FPS"]
        df = pd.DataFrame(all_metrics, columns=columns, index=[f"Seed_{i}" for i in range(1, 6)])

        """ Calculate mean and std """
        mean_metrics = df.mean(axis=0)
        std_metrics = df.std(axis=0)

        """ Save results to CSV """
        csv_path = f"results/metrics_summary_{model_name}.csv"
        df.loc["Mean"] = mean_metrics
        df.loc["Std"] = std_metrics
        df.to_csv(csv_path, index=True)

        """ Print summary """
        print(f"\nMetrics Summary for {model_name}:")
        print(df)
