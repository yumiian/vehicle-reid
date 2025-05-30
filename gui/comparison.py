import os
import cv2
import streamlit as st
import time

import database

def image_listdir(path):
    return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(('png', 'jpg', 'jpeg'))])

def label_listdir(path):
    return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(('txt'))])

def read_txt(txt_path):
    id_list = []
    xywh_dict = {}

    with open(txt_path, "r") as f:
        for line in f:
            id = line.rstrip().split(" ")[0]
            xywh = line.rstrip().split(" ")[1:]

            id_list.append(f"{int(id):06d}")
            xywh_dict[f"{int(id):06d}"] = xywh

    return (id_list, xywh_dict)

def draw_bbox(full_image, crop_path, id_list, xywh_dict):
    id = os.path.splitext(crop_path)[0].split("-")[-1]

    if id in id_list:
        xywh = xywh_dict[id]
        center_x = float(xywh[0])
        center_y = float(xywh[1])
        bbox_width = float(xywh[2])
        bbox_height = float(xywh[3])

        height, width, _ = full_image.shape

        x = center_x * width
        y = center_y * height
        w = bbox_width * width
        h = bbox_height * height

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        cv2.rectangle(full_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

def compare_images(crop1_path, crop2_path, image1_path, image2_path, label1_path, label2_path):
    if not (crop1_path and crop2_path):  # Ensure both images exist
        raise FileNotFoundError("One of the images is missing")
    
    crop1_id = os.path.splitext(os.path.basename(crop1_path))[0].split("-")[-1]
    crop2_id = os.path.splitext(os.path.basename(crop2_path))[0].split("-")[-1]
    image1_filename = os.path.splitext(os.path.basename(image1_path))[0]
    image2_filename = os.path.splitext(os.path.basename(image2_path))[0]
    
    if not st.session_state.display_first_only:
        img1_ = cv2.imread(crop1_path)
        img2_ = cv2.imread(crop2_path)
        img1 = cv2.cvtColor(img1_, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2_, cv2.COLOR_BGR2RGB)

        full_img1_ = cv2.imread(image1_path)
        full_img2_ = cv2.imread(image2_path)
        full_img1 = cv2.cvtColor(full_img1_, cv2.COLOR_BGR2RGB)
        full_img2 = cv2.cvtColor(full_img2_, cv2.COLOR_BGR2RGB)

        id_list1, xywh_dict1 = read_txt(label1_path)
        id_list2, xywh_dict2 = read_txt(label2_path)

        draw_bbox(full_img1, crop1_path, id_list1, xywh_dict1)
        draw_bbox(full_img2, crop2_path, id_list2, xywh_dict2)
        
        col1, col2 = st.columns(2, vertical_alignment="center", border=False)
        col3, col4 = st.columns(2, vertical_alignment="center", border=False)
        with col1:
            st.image(img1, caption=crop1_id, width=100)

        with col2:
            st.image(img2, caption=crop2_id, width=100)

        with col3:
            st.image(full_img1, caption=f"{image1_filename} ({st.session_state.img1} / {len(st.session_state.image_list1)})", width=500)

        with col4:
            st.image(full_img2, caption=f"{image2_filename} ({st.session_state.img2} / {len(st.session_state.image_list2)})", width=500)
    else:
        img1_ = cv2.imread(crop1_path)
        img1 = cv2.cvtColor(img1_, cv2.COLOR_BGR2RGB)

        full_img1_ = cv2.imread(image1_path)
        full_img1 = cv2.cvtColor(full_img1_, cv2.COLOR_BGR2RGB)

        id_list1, xywh_dict1 = read_txt(label1_path)

        draw_bbox(full_img1, crop1_path, id_list1, xywh_dict1)

        _, col1, _ = st.columns(3, vertical_alignment="center")
        _, col2, _ = st.columns(3, vertical_alignment="center")
        with col1:
            st.image(img1, caption=crop1_id, width=100)

        with col2:
            st.image(full_img1, caption=f"{image1_filename} ({st.session_state.img1} / {len(st.session_state.image_list1)})", width=500)

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

def save_results(save_path, results, mode="w"):
    with open(save_path, mode) as f:
        for entry in results:
            f.write(f"{entry['image1']} {entry['image2']}\n")

def get_image_label_path(img):
    path_list = os.path.normpath(img).split(os.path.sep) # ["gui","results","result","crops","KJ-C1-8AM-frame_000001-000000.jpg"]
    ori_path = os.path.join(*path_list[:-1]) # datasets2/track/KJ-C1-08AM/crops
    join_path = path_list[-1].split("-")[:-1] # ['KJ', 'C1', '08AM', 'frame_007489']
    new_join_path = "-".join(join_path) # KJ-C1-08AM-frame_007489
    full_path = os.path.join(ori_path, new_join_path) # datasets2/track/KJ-C1-08AM/crops\KJ-C1-08AM-frame_007489

    image_path = full_path + ".jpg" # datasets2/track/KJ-C1-08AM/crops\KJ-C1-08AM-frame_007489.jpg
    image_path = image_path.replace("crops", "images") # datasets2/track/KJ-C1-08AM/images\KJ-C1-08AM-frame_007489.jpg

    label_path = full_path + ".txt" # datasets2/track/KJ-C1-08AM/crops\KJ-C1-08AM-frame_007489.txt
    label_path = label_path.replace("crops", "labels") # datasets2/track/KJ-C1-08AM/labels\KJ-C1-08AM-frame_007489.txt

    # if path is in Windows (mnt), add slash in front to fix invalid path error
    if image_path.startswith("mnt"):
        image_path = "/" + image_path

    if label_path.startswith("mnt"):
        label_path = "/" + label_path

    return (image_path, label_path)

def save_last_imgs(resume_path, imgs1, imgs2):
    with open(resume_path, 'w') as f:
        for img1, img2 in zip(imgs1, imgs2):
            f.write(f"{img1} {img2}\n")

def read_last_imgs(resume_path):
    imgs1 = []
    imgs2 = []
    
    with open(resume_path, 'r') as f:
        for line in f:
            imgs = line.split()
            imgs1.append(imgs[0])
            imgs2.append(imgs[1])

    return (imgs1, imgs2)

def initialize_session_state():
    """Initialize all session state variables if they don't exist"""
    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    
    if "img1" not in st.session_state:
        st.session_state.img1 = 0
    if "img2" not in st.session_state:
        st.session_state.img2 = 0
    if "image_list1" not in st.session_state:
        st.session_state.image_list1 = []
    if "image_list2" not in st.session_state:
        st.session_state.image_list2 = []
    if 'undo_stack1' not in st.session_state:
        st.session_state.undo_stack1 = []
    if 'undo_stack2' not in st.session_state:
        st.session_state.undo_stack2 = []
    if "bback1_disabled" not in st.session_state:
        st.session_state.bback1_disabled = True
    if "bnext1_disabled" not in st.session_state:
        st.session_state.bnext1_disabled = False
    if "bback2_disabled" not in st.session_state:
        st.session_state.bback2_disabled = True
    if "bnext2_disabled" not in st.session_state:
        st.session_state.bnext2_disabled = False
    if "bmatch_disabled" not in st.session_state:
        st.session_state.bmatch_disabled = True

def resume_checkpoint():
    if not database.check_table_exist("checkpoint_info"):
        return
    
    if database.check_table_empty("checkpoint_info"):
        return
    
    st.session_state.crop_dir1 = database.select_data("checkpoint_info", "crop_dir1")[0][0] # access list of tuple [('data',)]
    st.session_state.crop_dir2 = database.select_data("checkpoint_info", "crop_dir2")[0][0]
    st.session_state.img1 = database.select_data("checkpoint_info", "index1")[0][0]
    st.session_state.img2 = database.select_data("checkpoint_info", "index2")[0][0]

def update_checkpoint():
    if database.check_table_exist("checkpoint_info"): # use the checkpoint saved directory instead of text input value
        st.session_state.crop_dir1 = database.select_data("checkpoint_info", "crop_dir1")[0][0]
        st.session_state.crop_dir2 = database.select_data("checkpoint_info", "crop_dir2")[0][0]

    database.drop_table("checkpoint_info")
    database.create_table("checkpoint_info")
    database.insert_data("checkpoint_info")

def start_comparison(keep_checkpoint=False, checkpoint=False):
    """Callback for the Run button"""
    try:
        if checkpoint:
            resume_checkpoint()
        # Initialize lists
        crop_files1 = image_listdir(st.session_state.crop_dir1)
        crop_files2 = image_listdir(st.session_state.crop_dir2)
        st.session_state.image_list1 = filter_files(crop_files1)
        st.session_state.image_list2 = filter_files(crop_files2)

        # create copy to prevent modify original image_list with NULL values
        datalist1 = st.session_state.image_list1.copy()
        datalist2 = st.session_state.image_list2.copy()
        database.create_table("comparison")
        database.insert_data("comparison", datalist1, datalist2)

        if not keep_checkpoint: # reset checkpoint
            database.drop_table("checkpoint")
            database.drop_table("checkpoint_info")

        st.session_state.is_running = True
    except FileNotFoundError:
        st.error("Invalid path!")
    except Exception as e:
        st.error(f"Something went wrong: {e}")

def next1():
    if len(st.session_state.image_list1) > 1:
        time.sleep(0.1)
        st.session_state.img1 += 1

def back1():
    if len(st.session_state.image_list1) > 1:
        time.sleep(0.1)
        st.session_state.img1 -= 1

def next2():
    if len(st.session_state.image_list2) > 1:
        time.sleep(0.1)
        st.session_state.img2 += 1

def back2():
    if len(st.session_state.image_list2) > 1:
        time.sleep(0.1)
        st.session_state.img2 -= 1

def del1():
    if len(st.session_state.image_list1) > 0:
        # Save the current state before deleting (push to undo stack)
        st.session_state.undo_stack1.append((
            st.session_state.image_list1[:],  # Clone the current list
            st.session_state.img1
        ))
        
        # delete image
        del st.session_state.image_list1[st.session_state.img1]
        st.session_state.img1 = min(st.session_state.img1, len(st.session_state.image_list1) - 1)

def del2():
    if len(st.session_state.image_list2) > 0:
        st.session_state.undo_stack2.append((
            st.session_state.image_list2[:], 
            st.session_state.img2
        ))

        del st.session_state.image_list2[st.session_state.img2]
        st.session_state.img2 = min(st.session_state.img2, len(st.session_state.image_list2) - 1) 

def undo_del1():
    if st.session_state.undo_stack1:
        # Pop the last state from the undo stack
        previous_state = st.session_state.undo_stack1.pop()
        st.session_state.image_list1, st.session_state.img1 = previous_state

def undo_del2():
    if st.session_state.undo_stack2:
        previous_state = st.session_state.undo_stack2.pop()
        st.session_state.image_list2, st.session_state.img2 = previous_state

def match():
    database.create_table("checkpoint")
    database.insert_data("checkpoint")

    update_checkpoint()
    
    # remove both images as they are matched
    del1()
    del2()

def undo_match():
    database.undo_last_match("checkpoint")

    update_checkpoint()

    undo_del1()
    undo_del2()

def save():
    if not database.check_table_exist("checkpoint"):
        st.error("No progress to be saved.")
        return False
    
    update_checkpoint()

    database.create_table("saved")
    database.copy_table("checkpoint", "saved", "id, image1, image2")
    st.success("Progress saved.")

    return True

def resume():
    if not database.check_table_exist("checkpoint"):
        st.error("No recent checkpoint to resume from.")
        return

    if database.check_table_empty("checkpoint"):
        st.error("No recent checkpoint to resume from.")
        return

    start_comparison(keep_checkpoint=True, checkpoint=True)
    filepath1 = database.compare_data("comparison", "checkpoint", "filepath1")
    filepath2 = database.compare_data("comparison", "checkpoint", "filepath2")
    st.session_state.image_list1 = [data["filepath1"] for data in filepath1]
    st.session_state.image_list2 = [data["filepath2"] for data in filepath2]
    st.success("Checkpoint resumed.")

def reset():
    if not database.check_table_exist("checkpoint"):
        st.error("No checkpoint found.")
        return
    
    database.drop_table("checkpoint")
    st.success("Checkpoint successfully deleted.")