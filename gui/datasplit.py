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

def create_txt(image_path, output_path, filename):
    with open(os.path.join(output_path, filename), "w") as f:
        for filename in os.listdir(image_path):
            if not filename.endswith(".jpg"):
                continue

            parts = filename.split('-')
            id = parts[-1].split('.')[0] 
            camera = parts[1]

            f.write(f"{filename} {id} {camera}\n")

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

def create_txt_labels(output_path):
    train_path = os.path.join(output_path, "train")
    gallery_path = os.path.join(output_path, "gallery")
    query_path = os.path.join(output_path, "query")

    create_txt(train_path, output_path, "train.txt")
    create_txt(gallery_path, output_path, "gallery.txt")
    create_txt(query_path, output_path, "query.txt")

def create_csv(image_path, output_path, image_type):
    csv_filename = image_type + ".csv"
    with open(os.path.join(output_path, csv_filename), "w") as f:
        f.write("path,id,camera\n")
        for filename in os.listdir(image_path):
            if not filename.endswith(".jpg"):
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
    # train__path = os.path.join(labels_path, "train_.csv")
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

def datasplit(crop_dir1, crop_dir2, output_path, split_ratio):
    crop_files1 = image_listdir(crop_dir1)
    crop_files2 = image_listdir(crop_dir2)

    crop_files = crop_files1 + crop_files2

    random.shuffle(crop_files)
    gallery = filter_files(crop_files)
    random.shuffle(crop_files)
    query = filter_files(crop_files)

    # Recursively split the dataset until no duplicates are found
    while not split(output_path, crop_files, gallery, query):
        # Re-shuffle files and regenerate gallery and query
        random.shuffle(crop_files)
        gallery = filter_files(crop_files)
        random.shuffle(crop_files)
        query = filter_files(crop_files)
        
    create_csv_labels(output_path)
    train_val_split(output_path, split_ratio)
    # create_txt_labels(output_path)

def transform_txt_labels(txt_path, out_path, img_dir):
    df = pd.read_csv(txt_path, sep=" ")
    df.columns = ["path", "id", "cam"]
    df = df[["path", "id"]]
    df["path"] = df["path"].apply(lambda x: os.path.join(img_dir, x))
    df.to_csv(out_path, index=False)

def train_val_csv_labels(train_txt_path, train_csv_path, val_csv_path):
    df = pd.read_csv(train_txt_path, sep=" ")
    df.columns = ["path", "id", "cam"]
    df = df[["path", "id"]]
    df["path"] = df["path"].apply(lambda x: os.path.join("train", x))

    # fix the seed to always generate the same sets
    random.seed(42)

    train_size = int(0.75 * len(df))
    val_size = len(df) - train_size
    val_idxes = random.sample(range(len(df)), val_size)
    train_idxes = list(set(range(len(df))) - set(val_idxes))

    train_df, val_df = df.loc[train_idxes], df.loc[val_idxes]
    len(train_df), len(val_df)
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)

def txt_to_csv_labels(labels_path):
    train_path = os.path.join(labels_path, "train.txt")
    gallery_path = os.path.join(labels_path, "gallery.txt")
    query_path = os.path.join(labels_path, "query.txt")
    val_path = os.path.join(labels_path, "val.csv")

    transform_txt_labels(gallery_path, os.path.splitext(gallery_path)[0] + ".csv", "gallery")
    transform_txt_labels(query_path, os.path.splitext(query_path)[0] + ".csv", "query")
    train_val_csv_labels(train_path, os.path.splitext(train_path)[0] + ".csv", val_path)