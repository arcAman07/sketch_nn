# Import the main class from the designer module
from .designer import NeuralNetworkDesigner

# Import any utility functions you want to make directly accessible
from .utils import save_uploaded_file  # assuming you have this function in utils.py

# You can also define the version of your package here
__version__ = "0.1.3"

# If you want to control what gets imported with "from sketch_nn import *"
__all__ = ['NeuralNetworkDesigner', 'save_uploaded_file']