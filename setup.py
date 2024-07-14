from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="sketch_nn",
    version="0.1.6",  # Incremented version number
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
        "pytesseract",
        "torch",
        "gradio",
        "fastapi",
        "uvicorn",
    ],
    author="Aman Sharma",
    author_email="amananytime07@gmail.com",
    description="A tool to generate PyTorch neural network code from flowchart images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={'sketch_nn': ['logos/*']},  # Include all files in logos directory
    include_package_data=True,
    url="https://github.com/arcAman07/sketch_nn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)