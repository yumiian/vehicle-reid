import csv
import os
import random
import pandas as pd
import subprocess

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

def veri_gallery(query_path, gallery_path, output_path):
    cmd = f"grep -v -F -f {query_path} {gallery_path} > {output_path}"
    
    subprocess.run(cmd, shell=True)