import os
import shutil

import database

def batch_rename(input_dir, output_dir, old, new):
    for filename in os.listdir(input_dir):
        if not filename.endswith(".jpg"):
            continue

        old_id = os.path.splitext(filename)[0].split("-")[-1]

        if old != old_id:
            continue
        
        old_filepath = os.path.join(input_dir, filename)
        new_filename = filename.replace(f"-{old}", f"-{new}")
        new_filepath = os.path.join(output_dir, new_filename)
        shutil.copy(old_filepath, new_filepath)

def read_result_file(result_file):
    imgs1 = []
    imgs2 = []
    
    with open(result_file, 'r') as f:
        for line in f:
            imgs = line.split()
            imgs1.append(imgs[0])
            imgs2.append(imgs[1])

    return (imgs1, imgs2)

def rename_files(dir1, dir2, output_dir1, output_dir2):
    # ids1, ids2 = read_result_file(result_file)

    image1 = database.select_data("saved", "image1")
    image2 = database.select_data("saved", "image2")
    ids1 = [id[0] for id in image1]
    ids2 = [id[0] for id in image2]

    for id1, id2 in zip(ids1, ids2):
        batch_rename(dir1, output_dir1, id1, id1)

    for id1, id2 in zip(ids1, ids2):
        batch_rename(dir2, output_dir2, id2, id1)