# Sketch NN

<div align="center">
  <img src="https://raw.githubusercontent.com/arcAman07/sketch_nn/master/logos/sketch_nn_logo.jpg" alt="Sketch NN Logo" width="200"/>
  <p><em>Transform hand-drawn neural network sketches into functional PyTorch code</em></p>
</div>

<div align="center">
  
  [![PyPI version](https://badge.fury.io/py/sketch-nn.svg)](https://badge.fury.io/py/sketch-nn)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python Versions](https://img.shields.io/pypi/pyversions/sketch-nn.svg)](https://pypi.org/project/sketch-nn/)

</div>

Sketch NN is an innovative Python library that transforms hand-drawn neural network sketches into functional PyTorch code. Design your neural architectures on paper, capture them with our tool, and watch as Sketch NN brings your ideas to life!

## 🌟 Features

- 📸 Process hand-drawn or digital sketches of neural network architectures
- 🧠 Support for a wide range of neural network layers
- 🔧 Generate ready-to-use PyTorch code
- 🖥️ User-friendly Gradio web interface for quick prototyping
- 🚀 FastAPI backend for scalable deployment

## 🛠️ Supported Layers

- Convolutional (Conv2D)
- Pooling (MaxPool2D, AvgPool2D)
- Fully Connected (Linear)
- Batch Normalization
- Dropout
- Activation Functions (ReLU, LeakyReLU, Sigmoid, Tanh)
- Recurrent (LSTM, GRU)
- Transformer
- Multi-head Attention

## 🚀 Installation

Install Sketch NN using pip:

```bash
pip install sketch_nn
```

## 🚀 Usage

```bash
from sketch_nn import NeuralNetworkDesigner

designer = NeuralNetworkDesigner()
designer.process_image('path_to_your_sketch.png')
pytorch_code = designer.generate_pytorch_code()
designer.write_to_file(pytorch_code, 'custom_nn.py')
```
## Gradio Web Interface Demo

```bash
from sketch_nn.demo import run_gradio_demo

run_gradio_demo()
```

