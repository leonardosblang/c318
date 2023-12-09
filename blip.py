import os
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess

def caption_images_in_folder(folder_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  
    model, vis_processors, _ = load_model_and_preprocess(name="blip_caption", model_type="base_coco", is_eval=True, device=device)


    for image_file in os.listdir(folder_path):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
           
            image_path = os.path.join(folder_path, image_file)
            raw_image = Image.open(image_path).convert("RGB")
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

  
            caption = model.generate({"image": image})[0]


            caption_file = os.path.splitext(image_file)[0] + ".txt"
            with open(os.path.join(folder_path, caption_file), 'w') as file:
                file.write(caption)


folder_path = 'dataset/'
caption_images_in_folder(folder_path)
