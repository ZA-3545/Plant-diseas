import torch
import torch.nn as nn
from torchvision import transforms, models
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

from fastapi.responses import HTMLResponse
app = FastAPI(title="Tomato Disease Classifier API")

@app.get("/upload", response_class=HTMLResponse)
def upload_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tomato Disease Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
            h1 { color: #2e7d32; }
            #dropzone { border: 2px dashed #ccc; padding: 40px; text-align: center; border-radius: 8px; cursor: pointer; }
            #dropzone.dragover { border-color: #2e7d32; background: #f1f8f1; }
            #preview { max-width: 300px; margin-top: 20px; border-radius: 8px; }
            #result { margin-top: 20px; padding: 15px; background: #f5f5f5; border-radius: 8px; }
            .top3-item { display: flex; justify-content: space-between; padding: 4px 0; }
            .loading { color: #888; }
        </style>
    </head>
    <body>
        <h1>🍅 Tomato Disease Classifier</h1>
        <div id="dropzone">
            <p>Image yahan drag karein, ya click kar ke select karein</p>
            <input type="file" id="fileInput" accept="image/*" style="display:none;">
        </div>
        <img id="preview" style="display:none;">
        <div id="result"></div>

        <script>
            const dropzone = document.getElementById('dropzone');
            const fileInput = document.getElementById('fileInput');
            const preview = document.getElementById('preview');
            const resultDiv = document.getElementById('result');

            dropzone.onclick = () => fileInput.click();

            dropzone.ondragover = (e) => { e.preventDefault(); dropzone.classList.add('dragover'); };
            dropzone.ondragleave = () => dropzone.classList.remove('dragover');
            dropzone.ondrop = (e) => {
                e.preventDefault();
                dropzone.classList.remove('dragover');
                handleFile(e.dataTransfer.files[0]);
            };

            fileInput.onchange = (e) => handleFile(e.target.files[0]);

            function handleFile(file) {
                if (!file) return;

                preview.src = URL.createObjectURL(file);
                preview.style.display = 'block';
                resultDiv.innerHTML = '<p class="loading">Predicting...</p>';

                const formData = new FormData();
                formData.append('file', file);

                fetch('/predict', { method: 'POST', body: formData })
                    .then(res => res.json())
                    .then(data => {
                        let html = `<h3>Predicted: ${data.predicted_class}</h3>`;
                        html += `<p>Confidence: ${(data.confidence * 100).toFixed(1)}%</p>`;
                        html += `<h4>Top 3:</h4>`;
                        data.top_3_predictions.forEach(p => {
                            html += `<div class="top3-item"><span>${p.class}</span><span>${(p.confidence*100).toFixed(1)}%</span></div>`;
                        });
                        resultDiv.innerHTML = html;
                    })
                    .catch(err => {
                        resultDiv.innerHTML = '<p style="color:red">Error: ' + err + '</p>';
                    });
            }
        </script>
    </body>
    </html>
    """



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = [
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite", "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus", "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]
num_classes = len(class_names)

# --- Model load karo (sirf ek baar, server start hone pe) ---
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load("finetuned_mobilenet_best.pth", map_location=device))
model = model.to(device)
model.eval()

# --- Image preprocessing (fine-tuned model ke training jaisa hi) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.get("/")
def root():
    return {"status": "Tomato Disease Classifier API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Image read karo
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Preprocess + predict
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

    predicted_class = class_names[predicted_idx.item()]

    # Top-3 predictions bhi dete hain (zyada useful info)
    top3_prob, top3_idx = torch.topk(probabilities, 3)
    top3 = [
        {"class": class_names[idx.item()], "confidence": round(prob.item(), 4)}
        for prob, idx in zip(top3_prob, top3_idx)
    ]

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence.item(), 4),
        "top_3_predictions": top3
    }