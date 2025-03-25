import os
import shutil

import database

def batch_rename(input_dir, output_dir, old, new):
    for filename in os.listdir(input_dir):
        if not filename.endswith(".png", ".jpg", ".jpeg"):
            continue

        old_id = os.path.splitext(filename)[0].split("-")[-1]

        if old != old_id:
            continue
        
        old_filepath = os.path.join(input_dir, filename)
        new_filename = filename.replace(f"-{old}", f"-{new}")
        new_filepath = os.path.join(output_dir, new_filename)
        shutil.copy(old_filepath, new_filepath)

def rename_files(dir1, dir2, output_dir1, output_dir2):
    # if not database.check_table_exist("saved"):
    #     st.error("Table does not exist!")
    #     return
    
    # if database.check_table_empty("saved"):
    #     st.error("No results to be saved!")
    #     return

    image1 = database.select_data("saved", "image1")
    image2 = database.select_data("saved", "image2")
    ids1 = [id[0] for id in image1]
    ids2 = [id[0] for id in image2]

    for id1, id2 in zip(ids1, ids2):
        batch_rename(dir1, output_dir1, id1, id1)

    for id1, id2 in zip(ids1, ids2):
        batch_rename(dir2, output_dir2, id2, id1)