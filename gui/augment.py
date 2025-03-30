import albumentations as A
import cv2
import os
import streamlit as st

def get_abbreviation(transform_list):
    """Generate abbreviation based on selected augmentations and probabilities."""
    abbr_map = {A.HorizontalFlip: "HF", A.VerticalFlip: "VF", A.RandomRotate90: "RR90", 
                A.Erasing: "ERS", A.ColorJitter: "CJ", A.Sharpen: "SHRP", 
                A.UnsharpMask: "UNSM", A.CLAHE: "CLAHE"}

    return "_".join(
        f"{abbr_map[type(t)]}{int(t.p * 100):02d}" for t in transform_list if type(t) in abbr_map
    ) + "_" if transform_list else ""

def augment(transform_list, seed, image_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    progress_text = st.empty()
    progress_bar = st.progress(0)
    total_files = len(os.listdir(image_path))

    transform = A.Compose(transform_list, seed=seed, strict=True)
    abbr = get_abbreviation(transform_list)

    for i, filename in enumerate(os.listdir(image_path), start=1):
        if not filename.endswith((".png", ".jpg", ".jpeg")):
            continue

        progress_text.text(f"Augmenting image file {i}/{total_files}: {filename} ({i/total_files*100:.2f}%)")

        filepath = os.path.join(image_path, filename)

        image = cv2.imread(filepath, cv2.COLOR_BGR2RGB)
        augmented_image = transform(image=image)["image"]
        
        output_filename = os.path.join(output_path, f"{abbr}{filename}")
        cv2.imwrite(output_filename, augmented_image)

        progress_bar.progress(i / total_files)

    progress_text.empty()
    progress_bar.empty()