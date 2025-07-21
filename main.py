import os
import cv2
import torch
from pathlib import Path
from network.model import Model
from PIL import Image
from app.neuro_app import main as app_main
import typer
import subprocess

device = "cuda" if torch.cuda.is_available() else "cpu"
RESHAPE = (128, 128)

def test_model(model, output_dir, arr, device, reshape):
    model.eval()
    with torch.no_grad():
        for i in range (len(arr)):
            image = cv2.imread(arr[i])
            image = cv2.resize(image, reshape)
            image = torch.tensor(image).float().permute(2, 0, 1) / 255.0  
            output_img = img_to_mask(model, image.to(device))
            output_img = Image.fromarray(output_img)
            output_img.save(os.path.join(output_dir, f"аутпут{i}.png"))
          
def img_to_mask(model, image):        
    output = torch.sigmoid(model(image))
    return output.squeeze().cpu().numpy() > 0.5 

        
model = Model("Unet", "resnet34", in_channels=3, out_classes=1)
app = typer.Typer()

@app.command()
def gui():
    app_main()

@app.command()
def train():
    subprocess.run(["python", "network/main.py"])
    

@app.command()
def run(model_file: str, in_dir: str, out_dir: str = "out"):
    model.load_state_dict(torch.load(model_file))

    ids = os.listdir(in_dir)
    images_filepaths = [os.path.join(in_dir, image_id) for image_id in ids]

    os.makedirs(out_dir, exist_ok=True)

    test_model(model, out_dir, images_filepaths, device, RESHAPE)

if __name__ == "__main__":
    app()