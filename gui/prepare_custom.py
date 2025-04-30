import csv
import os
import random
import numpy as np
import pandas as pd

def txt_to_csv(dataset_name, dataset_type, txt_path, csv_path):
    if dataset_name == "VeRi":
        with open(txt_path, "r") as file:
            lines = file.readlines()

        with open(csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            
            # Write the CSV header
            csv_writer.writerow(["path", "id", "camera"])
            
            # Process each line from the text file
            for line in lines:
                filename = line.strip()
                parts = filename.split("_")
                id_part = parts[0]  # "0002" 
                camera_part = parts[1]  # "c002" 

                # Write the row to the CSV
                path = os.path.join(dataset_type, filename)
                csv_writer.writerow([path, id_part, camera_part])

    if dataset_name == "VRIC":
        df = pd.read_csv(txt_path, sep=" ")
        df.columns = ["path", "id", "camera"]
        df["path"] = df["path"].apply(lambda x: os.path.join(dataset_type, x))

        df.to_csv(csv_path, index=False)

def train_val_split(dataset_dir, split_ratio, seed):
    train_path = os.path.join(dataset_dir, "train.csv")
    val_path = os.path.join(dataset_dir, "val.csv")

    df = pd.read_csv(train_path, dtype={"id": str}) # Ensure "id" is read as a string
    random.seed(seed)

    train_size = int(split_ratio * len(df))
    val_size = len(df) - train_size
    val_idxes = random.sample(range(len(df)), val_size)
    train_idxes = list(set(range(len(df))) - set(val_idxes))

    train_df, val_df = df.loc[train_idxes], df.loc[val_idxes]

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

def gallery_query_split(dataset_dir, random_state):
    rng = np.random.RandomState(random_state)

    test_path = os.path.join(dataset_dir, "test.csv")
    gallery_path = os.path.join(dataset_dir, "gallery.csv")
    query_path = os.path.join(dataset_dir, "query.csv")

    df = pd.read_csv(test_path, dtype={"id": str})
    test_ids = df["id"].unique()

    # build gallery & query from test IDs
    gallery_rows = []
    query_rows = []
    
    for ids in test_ids:
        group = df[df["id"] == ids]
        cams = group["camera"].unique()
        if len(cams) < 2:
            continue  # Skip IDs with insufficient camera views
        
        # Select query and gallery cameras
        query_cam = rng.choice(cams)
        query = group[group["camera"] == query_cam].sample(n=1, random_state=random_state)
        
        # Select ONE gallery image from a DIFFERENT camera
        other_cams = cams[cams != query_cam]
        gallery_cam = rng.choice(other_cams)
        gallery = group[group["camera"] == gallery_cam].sample(n=1, random_state=random_state)
        
        query_rows.append(query)
        gallery_rows.append(gallery)
    
    gallery_df = pd.concat(gallery_rows, ignore_index=True)
    query_df = pd.concat(query_rows, ignore_index=True)

    gallery_df.to_csv(gallery_path, index=False)
    query_df.to_csv(query_path, index=False)