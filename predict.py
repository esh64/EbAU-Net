import os
import time
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
import imageio  # Certifique-se de que esta biblioteca está instalada

from utils import create_dir, seeding

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)  ## (H, W, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (H, W, 3)
    return (mask * 255).astype(np.uint8)

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Criar diretório para resultados """
    create_dir("SegmentationResults")

    """ Carregar conjunto de testes """
    test_x = sorted(glob("data/test/image/*"))
    test_y = sorted(glob("data/test/mask/*"))

    """ Hyperparameters """
    H = 256
    W = 256
    size = (W, H)

    """ Modelos para teste """
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

        """ Importar módulo dinamicamente """
        imported_module = __import__(module_name)
        build_unet = imported_module.build_unet

        """ Carregar o modelo com Seed 4 """
        model_path = f"files/{model_name}_4.h5"
        model = build_unet()
        model = model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        """ Criar pasta para salvar resultados do modelo """
        model_results_dir = os.path.join("SegmentationResults", model_name)
        create_dir(model_results_dir)

        for i, x_path in tqdm(enumerate(test_x), total=len(test_x)):
            """ Ler imagem """
            image = cv2.imread(x_path, cv2.IMREAD_COLOR)  ## (H, W, 3)
            x = np.transpose(image, (2, 0, 1))  ## (3, H, W)
            x = x / 255.0
            x = np.expand_dims(x, axis=0)  ## (1, 3, H, W)
            x = torch.from_numpy(x).to(device, dtype=torch.float32)

            with torch.no_grad():
                """ Predição """
                pred_y = model(x)
                pred_y = torch.sigmoid(pred_y).cpu().numpy()[0, 0]  ## (H, W)
                pred_y = (pred_y > 0.5).astype(np.uint8)  ## Binarizar

            """ Salvar máscara segmentada """
            filename = os.path.basename(x_path)
            save_path = os.path.join(model_results_dir, filename)
            imageio.imwrite(save_path, mask_parse(pred_y))

        print(f"Máscaras segmentadas salvas em {model_results_dir}")

