from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from sketch_nn.designer import NeuralNetworkDesigner
from sketch_nn import utils
import tempfile
import os
import base64
import cv2
import numpy as np

app = FastAPI()

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Mount the static directory
static_dir = os.path.join(current_dir, "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_file = os.path.join(current_dir, "static", "index.html")
    if os.path.exists(html_file):
        with open(html_file, "r") as f:
            content = f.read()
        return HTMLResponse(content=content)
    else:
        return HTMLResponse(content="<h1>Error: index.html not found</h1>")

@app.post("/generate_nn/")
async def generate_nn(file: UploadFile = File(None), image_data: str = Form(None)):
    designer = NeuralNetworkDesigner()
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "input_image.png")
    
    if file:
        # Handle uploaded file
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
    elif image_data:
        # Handle base64 encoded image data from webcam
        image_data = image_data.split(",")[1]  # Remove the "data:image/png;base64," part
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        cv2.imwrite(temp_path, image)
    else:
        return {"error": "No image provided"}

    output_file = os.path.join(temp_dir, "custom_nn.py")
    designer.design_network(temp_path, output_file)
    return FileResponse(output_file, filename="custom_nn.py")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)