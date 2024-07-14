import os
import tempfile

def save_uploaded_file(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, "uploaded_image.png")
    uploaded_file.save(temp_path)
    return temp_path