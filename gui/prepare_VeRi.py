import csv
import os
import random
import pandas as pd

def txt_to_csv(txt_path, csv_path, dataset_type):
    with open(txt_path, 'r') as file:
        lines = file.readlines()

    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write the CSV header
        csv_writer.writerow(['path', 'id', 'camera'])
        
        # Process each line from the text file
        for line in lines:
            filename = line.strip()
            parts = filename.split('_')
            path = f"{dataset_type}/{filename}"
            id_part = parts[0]  # '0002' 
            camera_part = parts[1]  # 'c002' 

            # Write the row to the CSV
            csv_writer.writerow([path, id_part, camera_part])

def train_val_split(labels_path, split_ratio):
    train_path = os.path.join(labels_path, "train.csv")
    val_path = os.path.join(labels_path, "val.csv")

    df = pd.read_csv(train_path, dtype={'id': str}) # Ensure 'id' is read as a string
    random.seed(42)
    train_size = int(split_ratio * len(df))
    val_size = len(df) - train_size
    val_idxes = random.sample(range(len(df)), val_size)
    train_idxes = list(set(range(len(df))) - set(val_idxes))
    train_df, val_df = df.loc[train_idxes], df.loc[val_idxes]
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)