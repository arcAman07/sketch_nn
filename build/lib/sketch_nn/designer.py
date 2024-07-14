import cv2
import numpy as np
import pytesseract
import torch
import torch.nn as nn

class NeuralNetworkDesigner:
    def __init__(self):
        self.layer_maps = {}

    def process_image(self, image_path):
        # Read the image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold the image
        _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours from top to bottom
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
        
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            roi = gray[y:y+h, x:x+w]
            
            # Perform OCR on the ROI
            text = pytesseract.image_to_string(roi).strip()
            self.parse_layer_info(i, text)

    def parse_layer_info(self, layer_index, text):
        lines = text.split('\n')
        layer_info = {'type': 'Unknown', 'text': text}
        
        try:
            if 'Input' in lines[0]:
                layer_info['type'] = 'Input'
                layer_info['channels'] = int(lines[1]) if len(lines) > 1 else None
            elif 'Conv' in lines[0]:
                layer_info['type'] = 'Conv2d'
                layer_info['out_channels'] = int(lines[1]) if len(lines) > 1 else None
                layer_info['kernel_size'] = int(lines[2]) if len(lines) > 2 else None
            elif any(x in lines[0] for x in ['MaxPool', 'AvgPool']):
                layer_info['type'] = 'MaxPool2d' if 'Max' in lines[0] else 'AvgPool2d'
                layer_info['kernel_size'] = int(lines[1]) if len(lines) > 1 else None
            elif 'Linear' in lines[0]:
                layer_info['type'] = 'Linear'
                if len(lines) > 1 and '*' in lines[1]:
                    layer_info['in_features'] = lines[1]
                layer_info['out_features'] = int(lines[-1]) if lines[-1].isdigit() else None
            elif 'BatchNorm' in lines[0]:
                layer_info['type'] = 'BatchNorm2d'
                layer_info['num_features'] = int(lines[1]) if len(lines) > 1 else None
            elif any(x in lines[0] for x in ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']):
                layer_info['type'] = lines[0]
            elif 'Dropout' in lines[0]:
                layer_info['type'] = 'Dropout'
                layer_info['p'] = float(lines[1]) if len(lines) > 1 else 0.5
            elif 'Transformer' in lines[0]:
                layer_info['type'] = 'Transformer'
                layer_info['d_model'] = int(lines[1]) if len(lines) > 1 else 512
                layer_info['nhead'] = int(lines[2]) if len(lines) > 2 else 8
            elif 'Attention' in lines[0]:
                layer_info['type'] = 'MultiheadAttention'
                layer_info['embed_dim'] = int(lines[1]) if len(lines) > 1 else 512
                layer_info['num_heads'] = int(lines[2]) if len(lines) > 2 else 8
            elif 'LSTM' in lines[0] or 'GRU' in lines[0]:
                layer_info['type'] = lines[0]
                layer_info['hidden_size'] = int(lines[1]) if len(lines) > 1 else 256
                layer_info['num_layers'] = int(lines[2]) if len(lines) > 2 else 1
        except ValueError as e:
            print(f"Error parsing layer {layer_index}: {e}")

        self.layer_maps[layer_index] = layer_info
        print(f"Parsed layer {layer_index}: {layer_info}")  # Debug print

    def generate_pytorch_code(self):
        code = "import torch\nimport torch.nn as nn\n\n"
        code += "class CustomNN(nn.Module):\n"
        code += "    def __init__(self):\n"
        code += "        super(CustomNN, self).__init__()\n"
        
        forward_code = "    def forward(self, x):\n"
        
        in_channels = None
        for i, layer_info in sorted(self.layer_maps.items()):
            if layer_info['type'] == 'Input':
                in_channels = layer_info.get('channels', 3)
                continue
            
            if layer_info['type'] == 'Conv2d':
                out_channels = layer_info.get('out_channels', 64)
                kernel_size = layer_info.get('kernel_size', 3)
                code += f"        self.conv{i} = nn.Conv2d({in_channels}, {out_channels}, kernel_size={kernel_size}, padding=1)\n"
                forward_code += f"        x = self.conv{i}(x)\n"
                in_channels = out_channels
            
            elif layer_info['type'] in ['MaxPool2d', 'AvgPool2d']:
                kernel_size = layer_info.get('kernel_size', 2)
                code += f"        self.pool{i} = nn.{layer_info['type']}(kernel_size={kernel_size})\n"
                forward_code += f"        x = self.pool{i}(x)\n"
            
            elif layer_info['type'] == 'Linear':
                out_features = layer_info.get('out_features')
                if i == 1 or (i > 1 and self.layer_maps[i-1]['type'] not in ['Linear', 'Flatten']):
                    code += f"        self.flatten = nn.Flatten()\n"
                    forward_code += f"        x = self.flatten(x)\n"
                    in_features = layer_info.get('in_features', 'x.shape[1]')
                else:
                    in_features = self.layer_maps[i-1].get('out_features', 64)
                code += f"        self.fc{i} = nn.Linear({in_features}, {out_features})\n"
                forward_code += f"        x = self.fc{i}(x)\n"
            
            elif layer_info['type'] == 'BatchNorm2d':
                num_features = layer_info.get('num_features', in_channels)
                code += f"        self.bn{i} = nn.BatchNorm2d({num_features})\n"
                forward_code += f"        x = self.bn{i}(x)\n"
            
            elif layer_info['type'] in ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']:
                code += f"        self.act{i} = nn.{layer_info['type']}()\n"
                forward_code += f"        x = self.act{i}(x)\n"
            
            elif layer_info['type'] == 'Dropout':
                p = layer_info.get('p', 0.5)
                code += f"        self.dropout{i} = nn.Dropout(p={p})\n"
                forward_code += f"        x = self.dropout{i}(x)\n"
            
            elif layer_info['type'] == 'Transformer':
                d_model = layer_info.get('d_model', 512)
                nhead = layer_info.get('nhead', 8)
                code += f"        self.transformer{i} = nn.Transformer(d_model={d_model}, nhead={nhead})\n"
                forward_code += f"        x = self.transformer{i}(x)\n"
            
            elif layer_info['type'] == 'MultiheadAttention':
                embed_dim = layer_info.get('embed_dim', 512)
                num_heads = layer_info.get('num_heads', 8)
                code += f"        self.attention{i} = nn.MultiheadAttention(embed_dim={embed_dim}, num_heads={num_heads})\n"
                forward_code += f"        x, _ = self.attention{i}(x, x, x)\n"
            
            elif layer_info['type'] in ['LSTM', 'GRU']:
                hidden_size = layer_info.get('hidden_size', 256)
                num_layers = layer_info.get('num_layers', 1)
                code += f"        self.rnn{i} = nn.{layer_info['type']}(input_size={in_channels}, hidden_size={hidden_size}, num_layers={num_layers}, batch_first=True)\n"
                forward_code += f"        x, _ = self.rnn{i}(x)\n"
            
            elif layer_info['type'] == 'Unknown':
                print(f"Warning: Unknown layer type at index {i}. Layer info: {layer_info}")
        
        code += "\n" + forward_code
        code += "        return x\n"
        
        return code

    def write_to_file(self, code, filename):
        with open(filename, 'w') as f:
            f.write(code)

    def design_network(self, image_path, output_file):
        self.process_image(image_path)
        pytorch_code = self.generate_pytorch_code()
        self.write_to_file(pytorch_code, output_file)
        print(f"Neural network code has been generated and saved to '{output_file}'")
        print("\nGenerated PyTorch Code:")
        print(pytorch_code)