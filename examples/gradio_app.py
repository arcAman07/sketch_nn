import gradio as gr
from sketch_nn.designer import NeuralNetworkDesigner
import tempfile
import os
import cv2
import numpy as np

def generate_nn_code(image):
    designer = NeuralNetworkDesigner()
    
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "input_image.png")
    
    if isinstance(image, np.ndarray):
        # If it's a numpy array (captured image or uploaded image)
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        # If it's a file path (should not happen with current setup, but just in case)
        os.rename(image, temp_path)
    
    output_file = os.path.join(temp_dir, "custom_nn.py")
    designer.design_network(temp_path, output_file)
    with open(output_file, 'r') as f:
        code = f.read()
    return output_file, code

# Check if the version of Gradio supports the 'source' parameter
try:
    image_input = gr.Image(source=["upload", "webcam"], type="numpy", label="Upload or Capture Flowchart")
except TypeError:
    # Fallback for older Gradio versions
    image_input = gr.Image(type="numpy", label="Upload Flowchart")

iface = gr.Interface(
    fn=generate_nn_code,
    inputs=[image_input],
    outputs=[
        gr.File(label="Download Generated Code"),
        gr.Code(language="python", label="Generated PyTorch Code")
    ],
    title="Sketch NN: Neural Network Designer",
    description="Upload a flowchart image or capture one using your webcam to generate PyTorch code for your neural network architecture."
)

if __name__ == "__main__":
    iface.launch()