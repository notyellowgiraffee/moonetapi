import io
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List

app = FastAPI(title="MooNet Classification API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "moonet_final_scripted.pt"

species = ['Alambadi', 'Amritmahal', 'Ayrshire', 'Banni', 'Bargur', 'Bhadawari', 'Brown_Swiss', 'Dangi', 'Deoni', 'Gir', 'Guernsey', 'Hallikar', 'Hariana', 'Holstein_Friesian', 'Jaffrabadi', 'Jersey', 'Kangayam', 'Kankrej', 'Kasargod', 'Kenkatha', 'Kherigarh', 'Khillari', 'Krishna_Valley', 'Malnad_gidda', 'Mehsana', 'Murrah', 'Nagori', 'Nagpuri', 'Nili_Ravi', 'Nimari', 'Ongole', 'Pulikulam', 'Rathi', 'Red_Dane', 'Red_Sindhi', 'Sahiwal', 'Surti', 'Tharparkar', 'Toda', 'Umblachery', 'Vechur']

# Load TorchScript model (CPU only)
model = torch.jit.load(MODEL_PATH, map_location="cpu")
model.eval()

# Preprocessing (same as training)
preprocess = transforms.Compose([
    transforms.Resize((384, 384), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_top5(image_bytes: bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = preprocess(img).unsqueeze(0)  # shape (1,3,H,W)
    
    with torch.no_grad():
        preds = model(x)
        probs = torch.softmax(preds, dim=1)
        top_probs, top_idxs = probs.topk(5, dim=1)
        
    top_probs = top_probs.squeeze().tolist()
    top_idxs = top_idxs.squeeze().tolist()

    results = []
    for i in range(5):
        results.append({
            "species": species[top_idxs[i]],
            "confidence": round(top_probs[i] * 100, 2)
        })
    return results

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload an image file and get top 5 breed predictions.
    """
    image_bytes = await file.read()
    results = predict_top5(image_bytes)
    return {"prediction": results}

@app.get("/")
async def root():
    return {"message": "Mooo! Doja Cat *ppspsppssps*"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
