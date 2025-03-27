import os
import shutil
import random
import pandas as pd

def image_listdir(path):
    return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(('png', 'jpg', 'jpeg'))])

def filter_files(files):
    used_id = []
    unique_files = []

    for filename in files:
        id = os.path.splitext(filename)[0].split("-")[-1]

        if id in used_id:
            continue

        unique_files.append(filename)

        used_id.append(id)

    return unique_files

def move_files(input_file_list, output_path):
    for f in input_file_list:
        shutil.copy(f, os.path.join(output_path, os.path.basename(f)))

def split(output_path, files, gallery, query):
    train_path = os.path.join(output_path, "train")
    gallery_path = os.path.join(output_path, "gallery")
    query_path = os.path.join(output_path, "query")

    # make sure there is no duplicates on both lists
    if [x for x in gallery if x in query]:
        return False
    
    # remove gallery and query files from all files to create train files
    train = [y for y in files if y not in gallery and y not in query]

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(gallery_path, exist_ok=True)
    os.makedirs(query_path, exist_ok=True)

    move_files(train, train_path)
    move_files(gallery, gallery_path)
    move_files(query, query_path)

    # no duplicates found
    return True

def create_csv(image_path, output_path, image_type):
    csv_filename = image_type + ".csv"
    with open(os.path.join(output_path, csv_filename), "w") as f:
        f.write("path,id,camera\n")
        for filename in os.listdir(image_path):
            if not filename.endswith((".png", ".jpg", ".jpeg")):
                continue

            parts = filename.split('-')
            id = parts[-1].split('.')[0] 
            camera = parts[1]

            f.write(f"{image_type}/{filename},{id},{camera}\n")

def create_csv_labels(output_path):
    train_path = os.path.join(output_path, "train")
    gallery_path = os.path.join(output_path, "gallery")
    query_path = os.path.join(output_path, "query")

    create_csv(train_path, output_path, "train")
    create_csv(gallery_path, output_path, "gallery")
    create_csv(query_path, output_path, "query")

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

def datasplit(crop_dir, output_path, split_ratio, max_attempts=10):
    all_crop_files = []
    for dir in os.listdir(crop_dir):
        crop_files = image_listdir(os.path.join(crop_dir, dir))
        all_crop_files += crop_files

    if not all_crop_files: # check if crop files list is empty
        raise ValueError("No crop image files found in the specified directory.")
    
    for attempt in range(max_attempts):
        # Shuffle and create initial splits
        random.shuffle(all_crop_files)
        gallery = filter_files(all_crop_files.copy())
        random.shuffle(all_crop_files)
        query = filter_files(all_crop_files.copy())
        
        # Check if split is successful
        if split(output_path, all_crop_files, gallery, query):
            create_csv_labels(output_path)
            train_val_split(output_path, split_ratio)
            return
    
    # If max attempts reached without successful split
    raise ValueError(f"Unable to create a valid split after {max_attempts} attempts. Consider adjusting filtering or dataset.")