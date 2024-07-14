from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sketch_nn",
    version="0.1.4",
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
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    package_data={'sketch_nn': ['logos/plain_logo.jpg']},
    include_package_data=True,
    url="https://github.com/arcAman07/sketch_nn",
)