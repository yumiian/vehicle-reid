import os
import shutil
import streamlit as st

import database

def batch_rename(input_dir, output_dir, old_id, new_id):
    os.makedirs(output_dir, exist_ok=True)

    # check if the label id (old_id) is the same to exclude the unused label id
    valid_files = [
        filename for filename in os.listdir(input_dir) 
        if filename.endswith(('.png', '.jpg', '.jpeg')) and 
            os.path.splitext(filename)[0].split("-")[-1] == old_id 
    ]

    for filename in valid_files:
        old_filepath = os.path.join(input_dir, filename)
        new_filename = filename.replace(f"-{old_id}", f"-{new_id}")
        new_filepath = os.path.join(output_dir, new_filename)
        shutil.copy(old_filepath, new_filepath)

def loop_batch_rename(dir, output_dir, old_id, new_id, total_files, start_index):
    # progress bar
    progress_text = st.empty()
    progress_bar = st.progress(0)

    for i, (id1, id2) in enumerate(zip(old_id, new_id), start=start_index): 
        progress_text.write(f"Saving result {i}/{total_files} ({i/total_files*100:.2f}%)")
        batch_rename(dir, output_dir, id1, id2)
        progress_bar.progress(i / total_files)

    progress_text.empty()
    progress_bar.empty()

def rename_files(dir1, output_dir1, dir2=None, output_dir2=None):
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

    if dir2 and output_dir2:
        loop_batch_rename(dir=dir1, output_dir=output_dir1, old_id=ids1, new_id=ids1, total_files=len(ids1)+len(ids2), start_index=1) # rename first results
        loop_batch_rename(dir=dir2, output_dir=output_dir2, old_id=ids2, new_id=ids1, total_files=len(ids1)+len(ids2), start_index=len(ids1)+1) # rename second results
    else:
        loop_batch_rename(dir=dir1, output_dir=output_dir1, old_id=ids2, new_id=ids1, total_files=len(ids2), start_index=1) # rename second results only