import pytest
from sketch_nn.designer import NeuralNetworkDesigner
import os

@pytest.fixture
def designer():
    return NeuralNetworkDesigner()

def test_process_image(designer):
    # Assuming you have a test image in the tests directory
    test_image_path = os.path.join(os.path.dirname(__file__), 'test_flowchart.png')
    designer.process_image(test_image_path)
    assert len(designer.layer_maps) > 0

def test_parse_layer_info(designer):
    designer.parse_layer_info(0, "Conv2d\n64\n3")
    assert designer.layer_maps[0]['type'] == 'Conv2d'
    assert designer.layer_maps[0]['out_channels'] == 64
    assert designer.layer_maps[0]['kernel_size'] == 3

def test_generate_pytorch_code(designer):
    designer.layer_maps = {
        0: {'type': 'Input', 'channels': 3},
        1: {'type': 'Conv2d', 'out_channels': 64, 'kernel_size': 3},
        2: {'type': 'ReLU'},
        3: {'type': 'Linear', 'out_features': 10}
    }
    code = designer.generate_pytorch_code()
    assert 'class CustomNN(nn.Module):' in code
    assert 'self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)' in code
    assert 'self.act2 = nn.ReLU()' in code
    assert 'self.fc3 = nn.Linear(' in code