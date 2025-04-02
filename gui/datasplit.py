import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st

def get_all_images(images_dir):
    data = []

    for root, _, files in os.walk(images_dir):
        for file in files:
            if not file.endswith((".jpg", ".jpeg", ".png")):
                continue

            parts = file.split("-")
            if len(parts) >= 5:
                camera_id = parts[1]
                vehicle_id = parts[4].split(".")[0]
                img_path = os.path.join(root, file)
                data.append({
                    "path": img_path,
                    "id": vehicle_id,
                    "camera": camera_id
                })

    df = pd.DataFrame(data)

    return df

def subset_split(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    unique_id = df["id"].unique()

    train_val_ids, test_ids = train_test_split(unique_id, test_size=test_ratio, random_state=random_state)

    train_ids, val_ids = train_test_split(train_val_ids, test_size=val_ratio / (train_ratio + val_ratio), random_state=random_state)

    train_df = df[df["id"].isin(train_ids)]
    val_df = df[df["id"].isin(val_ids)]
    test_df = df[df["id"].isin(test_ids)]

    return train_df, val_df, test_df

def split_gallery_query(test_df, random_state=42):
    gallery_list = []
    query_list = []
    
    for _, group in test_df.groupby("id"):  
        if len(group["camera"].unique()) > 1:  
            query_sample = group.sample(n=1, random_state=random_state)  # Randomly select one sample as query
            query_list.append(query_sample)
            gallery_list.append(group.drop(query_sample.index))  # Remove query sample from gallery
        else:
            gallery_list.append(group) # If only one camera exists, assign all to gallery (avoiding empty gallery)

    gallery_df = pd.concat(gallery_list) if gallery_list else pd.DataFrame(columns=test_df.columns)
    query_df = pd.concat(query_list) if query_list else pd.DataFrame(columns=test_df.columns)

    return gallery_df, query_df

def copy_files(df, image_type, output_dir, progress_text, progress_bar, progress_now):
    output_subdir = os.path.join(output_dir, image_type)
    os.makedirs(output_subdir, exist_ok=True)

    source_path = df["path"].tolist()

    output_filename = [os.path.join(output_subdir, os.path.basename(path)) for path in source_path]

    total_files = len(source_path)

    for i, (src, dest) in enumerate(zip(source_path, output_filename), start=1):
        filename = os.path.basename(src)
        progress = progress_now + (0.2 * (i / total_files))
        progress_text.text(f"Copying files for {image_type} set {i}/{total_files}: {filename} ({progress*100:.2f}%)")
        shutil.copy(src, dest)
        progress_bar.progress(progress)
    
    # create CSV files
    image_filepath = [os.path.join(image_type, os.path.basename(path)) for path in source_path] # "query/KJ-C1-0800-frame_000018-000002.jpg"
    df["path"] = image_filepath # replace with new filepath 
    df.to_csv(os.path.join(output_dir, f"{image_type}.csv"), index=False)

def datasplit(images_dir, output_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    progress_text = st.empty()
    progress_bar = st.progress(0)

    progress_text.text("Scanning images... (0.00%)")
    progress_bar.progress(0)
    df = get_all_images(images_dir)

    progress_text.text("Splitting dataset into train, val, test set... (5.00%)")
    progress_bar.progress(0.05)
    train_df, val_df, test_df = subset_split(df, train_ratio, val_ratio, test_ratio, random_state)

    progress_text.text("Splitting dataset into gallery and query set... (10.00%)")
    progress_bar.progress(0.1)
    gallery_df, query_df = split_gallery_query(test_df, random_state)

    copy_files(train_df, "train", output_dir, progress_text, progress_bar, 0.2)
    copy_files(val_df, "val", output_dir, progress_text, progress_bar, 0.4)
    copy_files(gallery_df, "gallery", output_dir, progress_text, progress_bar, 0.6)
    copy_files(query_df, "query", output_dir, progress_text, progress_bar, 0.8)

    progress_text.empty()
    progress_bar.empty()