import os
import shutil
import streamlit as st

import database

def batch_rename(input_dir, output_dir, old, new):
    os.makedirs(output_dir, exist_ok=True)

    # check if the label id (old) is the same to exclude the unused label id
    valid_files = [
        filename for filename in os.listdir(input_dir) 
        if filename.endswith(('.png', '.jpg', '.jpeg')) and 
            os.path.splitext(filename)[0].split("-")[-1] == old 
    ]

    for filename in valid_files:
        old_filepath = os.path.join(input_dir, filename)
        new_filename = filename.replace(f"-{old}", f"-{new}")
        new_filepath = os.path.join(output_dir, new_filename)
        shutil.copy(old_filepath, new_filepath)

def rename_files(dir1, dir2, output_dir1, output_dir2):
    if not database.check_table_exist("saved"):
        st.error('Table "saved" does not exist!')
        return
    
    if database.check_table_empty("saved"):
        st.error("No results to be saved!")
        return

    image1 = database.select_data("saved", "image1")
    image2 = database.select_data("saved", "image2")

    ids1 = [id[0] for id in image1]
    ids2 = [id[0] for id in image2]

    # progress bar
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    total_files = len(ids1) + len(ids2)

    for i, (id1, id2) in enumerate(zip(ids1, ids2), start=1): 
        progress_text.write(f"Saving result {i}/{total_files} ({i/total_files*100:.2f}%)")
        batch_rename(dir1, output_dir1, id1, id1)
        progress_bar.progress(i / total_files)

    for i, (id1, id2) in enumerate(zip(ids1, ids2), start=len(ids1)+1): # continuing the index count
        progress_text.write(f"Saving result {i}/{total_files} ({i/total_files*100:.2f}%)")
        batch_rename(dir2, output_dir2, id2, id1)
        progress_bar.progress(i / total_files)

    progress_text.empty()
    progress_bar.empty()