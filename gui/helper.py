import os
import shutil

def cleanup(output_path):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

def save_file(file, output_path):
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "video.mp4"), "wb") as f:
        f.write(file)

def create_subfolders(base_path, folder_name):
    counter = 2
    while os.path.exists(os.path.join(base_path, folder_name)):
        folder_name = f"{folder_name}{counter}"
        counter += 1
    subfolder = os.path.join(base_path, folder_name)
    os.makedirs(subfolder)

    return subfolder
