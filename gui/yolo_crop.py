from ultralytics import YOLO
import cv2
import os
import shutil
import streamlit as st
import gc
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

import database

def extract_number(filename):
    parts = filename.split('_')
    
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

def edit_yolo_txt(input_path, output_path, verbose=False):
    with open(input_path, "r") as infile:
        lines = infile.readlines()

    with open(output_path, "w") as outfile:
        for line in lines:
            values = line.split()
            # Move the last int (unique id) to the first int (class)
            try:
                last_value = int(values[-1]) - 1  
                new_line = f"{last_value} " + " ".join(values[1:-1]) 
                outfile.write(new_line + "\n")
            except ValueError:
                if verbose:
                    print(f"Skipping line due to non-integer last value: {line.strip()}")
                pass # if it is not int, skip (bug)

def organize(save_path, prefix):
    labels_dir = os.path.join(save_path, "labels")
    os.makedirs(labels_dir, exist_ok=True)

    sorted_files = sorted(os.listdir(save_path), key=extract_number)
    valid_files = []
    for file in sorted_files:
        if "frame" not in file:
            continue

        label_path = os.path.join(save_path, file, "labels")
        label_txt = os.path.join(label_path, "image0.txt")

        if os.path.exists(label_txt):
            valid_files.append(file)

    total_files = len(valid_files)

    progress_text = st.empty()
    progress_bar = st.progress(0)

    def process_file(file):
        label_path = os.path.join(save_path, file, "labels")
        label_txt = os.path.join(label_path, "image0.txt")
        label_txt_new = os.path.join(label_path, "image.txt")

        # Extract ID safely
        id_part = file.split("_")[-1]
        id = id_part if id_part.isdigit() else "1"

        edit_yolo_txt(label_txt, label_txt_new)
        frame_txt = os.path.join(labels_dir, f"{prefix}-frame_{int(id):06d}"+".txt")
        os.rename(label_txt_new, frame_txt)
        
        return True
    
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_file, file): file
            for file in valid_files
        }
        
        for i, future in enumerate(as_completed(futures), start=1):
            filename = futures[future]
            progress_text.text(f"Organizing file {i}/{total_files}: {filename} ({i/total_files*100:.2f}%)")
            progress_bar.progress(i / total_files)

    progress_text.empty()
    progress_bar.empty()

def crop_labels(image_path, label_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    image = cv2.imread(image_path)
    img_h, img_w, _ = image.shape

    with open(label_path, "r") as f:
        lines = f.readlines()

    label_id_list = []

    for line in lines:
        values = line.split()
        id = values[0]
        x, y, w, h = values[1:]

        x = int(float(x) * img_w)
        y = int(float(y) * img_h)
        w = int(float(w) * img_w)
        h = int(float(h) * img_h)

        xmin = int(x - w/2)
        ymin = int(y - h/2)
        xmax = int(x + w/2)
        ymax = int(y + h/2)

        cropped = image[ymin:ymax, xmin:xmax]
        filename_noext = os.path.splitext(os.path.basename(image_path))[0]

        label_id = f"{int(id):06d}"
        label_id_list.append(label_id)
        output_filename = os.path.join(output_path, f"{filename_noext}-{label_id}.jpg")
        cv2.imwrite(output_filename, cropped)

    return label_id_list

def cleanup(save_path):
    main_files = {"images", "labels", "crops", "crops.txt"}
    all_files = os.listdir(save_path)
    files_to_del = [file for file in all_files if file not in main_files]
    total_files = len(files_to_del)

    progress_text = st.empty()
    progress_bar = st.progress(0)

    def delete_file(file):
        shutil.rmtree(os.path.join(save_path, file)) # remove unwanted files and folders

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(delete_file, file): file
            for file in files_to_del
        }
        
        for i, future in enumerate(as_completed(futures), start=1):
            filename = futures[future]
            progress_text.text(f"Cleaning folder {i}/{total_files}: {filename} ({i/total_files*100:.2f}%)")
            progress_bar.progress(i / total_files)

    progress_text.empty()
    progress_bar.empty()

def track(model, video_path, save_path, batchsize=100, conf=0.7, line_width=2, classes=[2, 7], save_frames=False, save_txt=False, prefix="result"):
    model = YOLO(model)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if save_frames:
        os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
    
    frame_count = 0

    progress_text = st.empty()
    progress_bar = st.progress(0)

    progress_text.text(f"Processing video with {total_frames} frames at {fps} FPS...")
    
    while cap.isOpened():
        # prevent memory buildup
        if frame_count % batchsize == 0 and frame_count > 0:
            gc.collect()
            torch.cuda.empty_cache()

        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        model.track(source=frame, persist=True, conf=conf, line_width=line_width, classes=classes, project=save_path, name=f"{prefix}-frame_{frame_count:06d}", save_txt=save_txt, verbose=False)
        
        if save_frames:
            frame_filename = f'{prefix}-frame_{frame_count:06d}.jpg'
            frame_path = os.path.join(save_path, "images", frame_filename)

            if not os.path.exists(frame_path):
                cv2.imwrite(frame_path, frame)
        
        progress_text.text(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.2f}%)")
        progress_bar.progress(frame_count / total_frames)
    
    cap.release()

    progress_text.empty()
    progress_bar.empty()
    
    # clean up
    if save_frames or save_txt:
        organize(save_path, prefix)
    cleanup(save_path)

    gc.collect()
    torch.cuda.empty_cache()

def save_crop(save_path):
    label_path = os.path.join(save_path, "labels")
    image_path = os.path.join(save_path, "images")
    crop_path = os.path.join(save_path, "crops")

    if not os.path.exists(label_path):
        raise FileNotFoundError("labels folder not found")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError("images folder not found")

    label_files = os.listdir(label_path)
    total_files = len(label_files)
    
    progress_text = st.empty()
    progress_bar = st.progress(0)

    database.create_table("image")
    database.create_table("crop_image")

    frame_label_list = []

    def process_file(file):
        label_file = os.path.join(label_path, file)
        image_file = os.path.join(image_path, os.path.splitext(file)[0] + ".jpg")
        frame_id = os.path.splitext(file)[0].split("-")[-1].split("_")[-1]
        label_ids = crop_labels(image_file, label_file, crop_path)
        
        return frame_id, label_ids
    
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_file, file): file
            for file in label_files
        }
        
        for i, future in enumerate(as_completed(futures), start=1):
            filename = futures[future]
            frame_id, label_ids = future.result()
            frame_label_list.append((frame_id, label_ids))
            progress_text.text(f"Creating crop image file {i}/{total_files}: {filename} ({i/total_files*100:.2f}%)")
            progress_bar.progress(i / total_files)

    database.insert_data("image", frame_label_list)
    database.insert_data("crop_image", frame_label_list)
    
    progress_text.empty()
    progress_bar.empty()