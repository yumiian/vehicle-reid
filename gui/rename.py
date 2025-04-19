import os
import shutil
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

import database

def batch_rename(input_dir, output_dir, old_id, new_id):
    os.makedirs(output_dir, exist_ok=True)

    # check if the label id (old_id) is the same to exclude the unused label id
    valid_files = [
        filename for filename in os.listdir(input_dir) 
        if filename.endswith(('.png', '.jpg', '.jpeg')) and 
            os.path.splitext(filename)[0].split("-")[-1] == old_id 
    ]

    def process_file(filename):
        old_filepath = os.path.join(input_dir, filename)
        new_filename = filename.replace(f"-{old_id}", f"-{new_id}")
        new_filepath = os.path.join(output_dir, new_filename)
        shutil.copy(old_filepath, new_filepath)

    with ThreadPoolExecutor() as executor:
        executor.map(process_file, valid_files)

def process_all_renames(dir1, output_dir1, dir2, output_dir2, ids1, ids2):
    total_files = len(ids1) + (len(ids2) if dir2 else 0) # both ids have same len

    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    with ThreadPoolExecutor() as executor:
        futures = []

        for id1, id2 in zip(ids1, ids2):
            if dir2 and output_dir2:
                futures.append(executor.submit(batch_rename, dir1, output_dir1, id1, id1)) # rename first results
                futures.append(executor.submit(batch_rename, dir2, output_dir2, id2, id1)) # rename second results
            else:
                if not st.session_state.display_first_only:
                    futures.append(executor.submit(batch_rename, dir1, output_dir1, id2, id1)) # rename second results only
                else:
                    futures.append(executor.submit(batch_rename, dir1, output_dir1, id1, id1)) # rename first results only
        
        for i, future in enumerate(as_completed(futures), start=1):
            progress = i / total_files
            progress_text.write(f"Saving result {i}/{total_files} ({progress*100:.2f}%)")
            progress_bar.progress(progress)
    
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

    process_all_renames(dir1, output_dir1, dir2, output_dir2, ids1, ids2)