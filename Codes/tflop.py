import torch
import os
from flopth import flopth

# Lista de arquivos dos modelos
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


if __name__ == "__main__":
    for model_file in model_files:
        module_name = model_file.replace(".py", "")  # Remove extensão .py
        model_name = module_name.replace("model", "")  # Nome do modelo sem prefixo "model"
        print(f"Testing {model_name}")
        
        # Importa o módulo dinamicamente
        imported_module = __import__(module_name)
        build_unet = imported_module.build_unet  
        model = build_unet()
        
        dummy_inputs = torch.rand(5, 3, 256, 256)
        flops, params = flopth(model, inputs=(dummy_inputs,))
        print(f"The model {model_name} have: {flops} TFLOPs and {params} params.")
