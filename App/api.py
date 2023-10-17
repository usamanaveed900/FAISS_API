import os
import pickle
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi import File
from fastapi import UploadFile
from fastapi import Form
import torch
import torch.nn as nn
from torchvision import transforms, models
from image_processor import ImagePrep  # Import your image processing script here
import faiss
import json
import numpy as np


class FeatureExtractor(nn.Module):
    def __init__(self,
                 decoder: dict = None):
        super(FeatureExtractor, self).__init__()
        self.model    = models.resnet50(weights='IMAGENET1K_V2')
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 13)
        self.model.eval()
        
        self.decoder = decoder

    def forward(self, image):
        x = self.model(image)
        return x
    def predict(self, image):
        with torch.no_grad():
            x = self.forward(image)
            return x

# Don't change this, it will be useful for one of the methods in the API
class TextItem(BaseModel):
    text: str

try:
    # Load the Feature Extraction model
    model = FeatureExtractor(decoder={"Home & Garden": 0, "Baby & Kids Stuff": 1, "DIY Tools & Materials": 2, "Music, Films, Books & Games": 3, "Phones, Mobile Phones & Telecoms": 4, "Clothes, Footwear & Accessories": 5, "Other Goods": 6, "Health & Beauty": 7, "Sports, Leisure & Travel": 8, "Appliances": 9, "Computers & Software": 10, "Office Furniture & Equipment": 11, "Video Games & Consoles": 12})  # Initialize with your decoder
    model.load_state_dict(torch.load('final_model/image_model.pth', map_location=torch.device('cpu')))
    model.eval()
except:
    raise OSError("No Feature Extraction model found. Check that you have the decoder and the model in the correct location")

try:
    # Load the FAISS index and image IDs
    with open("image_embeddings.json", 'r') as f:
        image_embeddings = json.load(f)
    
    index = faiss.read_index("FAISS_index.pkl")
except:
    raise OSError("No Image model found. Check that you have the encoder and the model in the correct location")

app = FastAPI()
print("Starting server")

@app.get('/healthcheck')
def healthcheck():
  msg = "API is up and running!"
  
  return {"message": msg}

@app.post('/predict/feature_embedding')
def predict_image(image: UploadFile = File(...)):
    contents = image.file.read()

    with open(image.filename, 'wb') as f:
        f.write(contents) 
    with open(image.filename, 'rb') as f:
        pil_image = Image.open(image.filename)

    ip         = ImagePrep(pil_image)
    tens_image = ip.img
    model_input = tens_image.to(torch.device('cpu'))
    img_emb    = model.forward(model_input)
    os.remove(image.filename)

    return JSONResponse(content={
                                "features": img_emb.tolist()[0], 
                                    })

@app.post('/predict/similar_images')
def predict_combined(image: UploadFile = File(...)):
    contents = image.file.read()
    with open(image.filename, 'wb') as f:
        f.write(contents) 
    with open(image.filename, 'rb') as f:
        pil_image = Image.open(image.filename)

    tens_image = ImagePrep(pil_image).img
    with torch.no_grad():
        img_emb = model.forward(tens_image)  # Use the model inside FeatureExtractor for predictions

    _, I = index.search(img_emb.numpy(), 4)  # Perform the search
    os.remove(image.filename)

    key = list(image_embeddings.keys())
    img_labels = []
    for _ in I.tolist()[0]:
        img_labels.append(key[_])

    return JSONResponse(content={
                                "similar_index": I.tolist()[0], # Return the index of similar images here
                                "image_labels": img_labels
                                })


if __name__ == '__main__':
    uvicorn.run("api:app", host="0.0.0.0", port=8080)
