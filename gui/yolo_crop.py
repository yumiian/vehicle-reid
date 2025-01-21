from ultralytics import YOLO
import cv2
import os
import shutil

def create_txt(image_path, output_path, filename):
    with open(os.path.join(output_path, filename), "w") as f:
        for filename in os.listdir(image_path):
            if not filename.endswith(".jpg"):
                continue

            parts = filename.split('-')
            id = parts[-1].split('.')[0] 
            camera = parts[1]

            f.write(f"{filename} {id} {camera}\n")

def extract_number(filename):
    parts = filename.split('_')
    
    return int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0

def edit_yolo_txt(input_path, output_path):
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
                print(f"Skipping line due to non-integer last value: {line.strip()}")
                pass # if it is not int, skip (bug)

def organize(save_path, prefix):
    sorted_files = sorted(os.listdir(save_path), key=extract_number)

    for file in sorted_files:
        # skip the folder name that is not frame
        if "frame" not in file:
            continue

        id = file.split("_")[-1]
        if id == "":
            id = 1

        label_path = os.path.join(save_path, file, "labels")
        label_txt = os.path.join(label_path, "image0.txt")
        label_txt_new = os.path.join(label_path, "image.txt")

        # if there is no label, skip
        if not os.path.exists(label_txt):
            continue
        
        edit_yolo_txt(label_txt, label_txt_new)
        os.makedirs(os.path.join(save_path, "labels"), exist_ok=True)
        frame_txt = os.path.join(save_path, "labels", f"{prefix}-frame_{int(id):06d}"+".txt")
        os.rename(label_txt_new, frame_txt)

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

        output_filename = os.path.join(output_path, filename_noext + f"-{int(id):06d}.jpg")
        cv2.imwrite(output_filename, cropped)

def cleanup(save_path):
    main_folder = ["images", "labels", "crops", "crops.txt"]

    for folder in os.listdir(save_path):
        if folder not in main_folder:
            shutil.rmtree(os.path.join(save_path, folder)) # remove unwanted folders

def track(model, video_path, save_path, show=False, conf=0.7, line_width=2, classes=[2, 7], save_frames=False, save_txt=False, prefix="result"):
    model = YOLO(model)
    cap = cv2.VideoCapture(video_path)

    filename = f"{prefix}-frame_"

    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break
        
        results = model.track(frame, persist=True, conf=conf, line_width=line_width, classes=classes, project=save_path, name=filename, save_txt=save_txt)

        # Visualize the results on the frame
        if show:
            annotated_frame = results[0].plot()
            cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Save frames
        if save_frames:
            frame_filename = f'{prefix}-frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES)):06d}.jpg'
            os.makedirs(os.path.join(save_path, "images"), exist_ok=True)
            frame_path = os.path.join(save_path, "images", frame_filename)
            if not os.path.exists(frame_path):
                cv2.imwrite(frame_path, frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Clean up
    if save_frames or save_txt:
        organize(save_path, prefix)

    # cleanup
    cleanup(save_path)

def save_crop(save_path):
    label_path = os.path.join(save_path, "labels")
    image_path = os.path.join(save_path, "images")
    crop_path = os.path.join(save_path, "crops")

    if not os.path.exists(label_path):
        raise FileNotFoundError("labels folder not found")
    
    if not os.path.exists(image_path):
        raise FileNotFoundError("images folder not found")

    for file in os.listdir(label_path):
        label_file = os.path.join(label_path, file)
        image_file = os.path.join(image_path, os.path.splitext(file)[0] + ".jpg")

        if not os.path.exists(image_file):
            continue

        crop_labels(image_file, label_file, crop_path)

    create_txt(crop_path, os.path.dirname(crop_path), "crops.txt")

