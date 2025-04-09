import os
import shutil
import numpy as np
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

def subset_split(df, train_ratio, val_ratio, test_ratio, random_state):
    rng = np.random.RandomState(random_state)

    # drop IDs with too few images
    sizes = df.groupby("id").size()
    good_ids = sizes[sizes >= 2].index
    df = df[df["id"].isin(good_ids)].reset_index(drop=True)
    
    # all unique IDs
    all_ids = df["id"].unique()
    train_val_ids, test_ids = train_test_split(all_ids, test_size=test_ratio, random_state=random_state)
    
    # build train & val by sampling per ID
    train_rows = []
    val_rows = []
    for ids in train_val_ids:
        group = df[df["id"] == ids]
        n_val = int(len(group) * (val_ratio / (train_ratio + val_ratio)))
        n_val = max(1, min(n_val, len(group) - 1)) # ensure at least 1, at most len(group)-1
        val_idx = group.sample(n=n_val, random_state=random_state).index

        train_rows.append(group.drop(val_idx))
        val_rows.append(group.loc[val_idx])
    
    train_df = pd.concat(train_rows, ignore_index=True)
    val_df = pd.concat(val_rows, ignore_index=True)
    
    # build gallery & query from test IDs
    gallery_rows = []
    query_rows = []
    for ids, group in df[df["id"].isin(test_ids)].groupby("id"):
        cams = group["camera"].unique()
        if len(cams) < 2:
            continue # require at least 2 cameras to form a cross-camera query/gallery
        
        query_cam = rng.choice(cams) # pick one image from one camera as query
        query = group[group["camera"] == query_cam].sample(n=1, random_state=random_state) # sample one image from that camera
        gallery = group.drop(query.index)
        
        gallery_rows.append(gallery)
        query_rows.append(query)
    
    gallery_df = pd.concat(gallery_rows, ignore_index=True)
    query_df = pd.concat(query_rows, ignore_index=True)
    
    return train_df, val_df, gallery_df, query_df

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

    progress_text.text("Splitting dataset into train, val, gallery and query sets... (10.00%)")
    progress_bar.progress(0.1)
    train_df, val_df, gallery_df, query_df = subset_split(df, train_ratio, val_ratio, test_ratio, random_state)

    copy_files(train_df, "train", output_dir, progress_text, progress_bar, 0.2)
    copy_files(val_df, "val", output_dir, progress_text, progress_bar, 0.4)
    copy_files(gallery_df, "gallery", output_dir, progress_text, progress_bar, 0.6)
    copy_files(query_df, "query", output_dir, progress_text, progress_bar, 0.8)

    progress_text.empty()
    progress_bar.empty()