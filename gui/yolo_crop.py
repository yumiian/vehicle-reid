from ultralytics import YOLO
import cv2
import os
import shutil
import streamlit as st
import gc
import time

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
    sorted_files = sorted(os.listdir(save_path), key=extract_number)
    total_files = len(sorted_files)

    progress_text = st.empty()
    progress_bar = st.progress(0)

    for i, file in enumerate(sorted_files, start=1):
        # skip the folder name that is not frame
        if "frame" not in file:
            continue

        label_path = os.path.join(save_path, file, "labels")
        label_txt = os.path.join(label_path, "image0.txt")
        label_txt_new = os.path.join(label_path, "image.txt")

        # if there is no label, skip
        if not os.path.exists(label_txt):
            continue

        progress_text.text(f"Organizing file {i}/{total_files}: {file} ({i/total_files*100:.2f}%)")

        id = file.split("_")[-1]
        if id == "":
            id = 1
        
        edit_yolo_txt(label_txt, label_txt_new)
        os.makedirs(os.path.join(save_path, "labels"), exist_ok=True)
        frame_txt = os.path.join(save_path, "labels", f"{prefix}-frame_{int(id):06d}"+".txt")
        os.rename(label_txt_new, frame_txt)

        progress_bar.progress(i / total_files)

    progress_text.empty()
    progress_bar.empty()

def crop_labels(image_path, label_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    image = cv2.imread(image_path)
    img_h, img_w, _ = image.shape

    with open(label_path, "r") as f:
        lines = f.readlines()

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
        output_filename = os.path.join(output_path, f"{filename_noext}-{label_id}.jpg")
        cv2.imwrite(output_filename, cropped)

        # database
        database.create_table("crop_image")
        database.insert_data("crop_image", label_id) 

def cleanup(save_path):
    main_folder = ["images", "labels", "crops", "crops.txt"]

    for folder in os.listdir(save_path):
        if folder not in main_folder:
            shutil.rmtree(os.path.join(save_path, folder)) # remove unwanted folders

def track(model, video_path, save_path, batchsize=100, conf=0.7, line_width=2, classes=[2, 7], save_frames=False, save_txt=False, prefix="result"):
    model = YOLO(model)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if save_frames:
        os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
    
    # Process in batches to prevent memory leaks
    frame_count = 0

    # Streamlit
    progress_text = st.empty()
    progress_bar = st.progress(0)
    stats_container = st.empty()
    progress_text.text(f"Processing video with {total_frames} frames at {fps} FPS...")

    start_time = time.time()
    processing_times = []
    
    while cap.isOpened():
        # Reset model state periodically to prevent memory buildup
        if frame_count % batchsize == 0 and frame_count > 0:
            gc.collect() # Force garbage collection to free memory
            
            # Reinitialize the tracker state
            if hasattr(model, "reset_tracker"):
                model.reset_tracker()
        
        # Measure frame processing time
        frame_start_time = time.time()

        # Read the next frame
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        
        results = model.track(source=frame, persist=True, conf=conf, line_width=line_width, classes=classes, project=save_path, 
                              name=f"{prefix}-frame_{frame_count:06d}", save_txt=save_txt, verbose=False)
        
        if save_frames:
            frame_filename = f'{prefix}-frame_{frame_count:06d}.jpg'
            frame_path = os.path.join(save_path, "images", frame_filename)
            # check if image file already exists
            if not os.path.exists(frame_path):
                # Use the annotated frame from results if available
                if hasattr(results, 'plot') and results[0].plot is not None:
                    plotted_frame = results[0].plot()
                    cv2.imwrite(frame_path, plotted_frame)
                else:
                    cv2.imwrite(frame_path, frame)
        
        # Calculate frame processing time
        frame_time = time.time() - frame_start_time
        processing_times.append(frame_time)
        avg_time = sum(processing_times[-batchsize:]) / min(len(processing_times), batchsize)  # Moving average
        
        # Update Streamlit progress
        if st and frame_count % 10 == 0:  
            progress_text.text(f"Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.2f}%)")
            progress_bar.progress(frame_count / total_frames)
            
            elapsed_time = time.time() - start_time
            estimated_total = (elapsed_time / frame_count) * total_frames
            remaining_time = estimated_total - elapsed_time
            minutes, seconds = map(int, divmod(elapsed_time, 60))
            rem_min, rem_s = map(int, divmod(remaining_time, 60))
            
            stats_container.write(f"""
            Current Statistics:
            | Metric | Value |
            | --- | --- |
            | Elapsed Time | {minutes:02d}:{seconds:02d} |
            | Average Processing Time | {avg_time * 1000:.2f} ms/frame |
            | Estimated Time Remaining | {rem_min:02d}:{rem_s:02d} |
            | Current FPS | {frame_count / elapsed_time:.2f} |
            """)
    
    cap.release()

    progress_text.empty()
    progress_bar.empty()
    stats_container.empty()

    total_time = time.time() - start_time
    total_min, total_s = map(int, divmod(total_time, 60))

    st.write("Overall Statistics:")
    st.write(f"""
            | Metric | Value |
            | --- | --- |
            | Total Frames Processed | {frame_count} |
            | Total Time Taken | {total_min:02d}:{total_s:02d} |
            | Average FPS | {frame_count / (total_time):.2f} |
            """)
    
    # Clean up
    if save_frames or save_txt:
        organize(save_path, prefix)
    cleanup(save_path)

def save_crop(save_path):
    label_path = os.path.join(save_path, "labels")
    image_path = os.path.join(save_path, "images")
    crop_path = os.path.join(save_path, "crops")

    if not os.path.exists(label_path):
        raise FileNotFoundError("labels folder not found")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError("images folder not found")

    # Get list of files first to determine total progress
    label_files = os.listdir(label_path)
    total_files = len(label_files)
    
    # Create a progress bar
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    # Process each file with progress updates
    for i, file in enumerate(label_files, start=1):
        # Update status text
        progress_text.text(f"Creating crop image file {i}/{total_files}: {file} ({i/total_files*100:.2f}%)")
        
        label_file = os.path.join(label_path, file)
        image_file = os.path.join(image_path, os.path.splitext(file)[0] + ".jpg")

        if not os.path.exists(image_file):
            continue

        # database
        database.create_table("image")
        database.insert_data("image", os.path.splitext(file)[0].split("-")[-1].split("_")[-1]) # frame_000000 -> 000000

        crop_labels(image_file, label_file, crop_path)
        
        # Update progress bar
        progress_bar.progress(i / total_files)
    
    progress_text.empty()
    progress_bar.empty()