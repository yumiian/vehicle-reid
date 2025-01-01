import os
import cv2
import streamlit as st

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
    
    img1_ = cv2.imread(crop1_path)
    img2_ = cv2.imread(crop2_path)
    img1 = cv2.cvtColor(img1_, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2_, cv2.COLOR_BGR2RGB)
    # img1_resized = cv2.resize(img1, None, fx=0.7, fy=0.7)
    # img2_resized = cv2.resize(img2, None, fx=0.7, fy=0.7)

    full_img1_ = cv2.imread(image1_path)
    full_img2_ = cv2.imread(image2_path)
    full_img1 = cv2.cvtColor(full_img1_, cv2.COLOR_BGR2RGB)
    full_img2 = cv2.cvtColor(full_img2_, cv2.COLOR_BGR2RGB)
    # full_img1_resized = cv2.resize(full_img1, None, fx=0.7, fy=0.7)
    # full_img2_resized = cv2.resize(full_img2, None, fx=0.7, fy=0.7)

    id_list1, xywh_dict1 = read_txt(label1_path)
    id_list2, xywh_dict2 = read_txt(label2_path)

    draw_bbox(full_img1, crop1_path, id_list1, xywh_dict1)
    draw_bbox(full_img2, crop2_path, id_list2, xywh_dict2)
    
    col1, col2 = st.columns(2, vertical_alignment="center", border=False)
    col3, col4 = st.columns(2, vertical_alignment="center", border=False)
    # with st.container(border=True):
    with col1:
        # st.header("Crop 1")
        st.image(img1, caption="Crop 1", width=100)
        # st.image(img1_resized, caption="Crop 1")

    with col2:
        # st.header("Crop 2")
        st.image(img2, caption="Crop 2", width=100)
        # st.image(img2_resized, caption="Crop 2")

    with col3:
        # st.header("Image 1")
        st.image(full_img1, caption="Full image 1", width=550)
        # st.image(full_img1_resized, caption="Full image 1")

    with col4:
        # st.header("Image 2")
        st.image(full_img2, caption="Full image 2", width=550)
        # st.image(full_img2_resized, caption="Full image 2")

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
    # delimiter = os.path.splitext(img)[0].split("\\")
    # ori_path = delimiter[0]  # datasets2/track/KJ-C1-08AM/crops
    # join_path = delimiter[-1].split("-")[:-1] # ['KJ', 'C1', '08AM', 'frame_007489']
    new_join_path = "-".join(join_path) # KJ-C1-08AM-frame_007489
    full_path = os.path.join(ori_path, new_join_path) # datasets2/track/KJ-C1-08AM/crops\KJ-C1-08AM-frame_007489

    image_path = full_path + ".jpg" # datasets2/track/KJ-C1-08AM/crops\KJ-C1-08AM-frame_007489.jpg
    image_path = image_path.replace("crops", "images") # datasets2/track/KJ-C1-08AM/images\KJ-C1-08AM-frame_007489.jpg

    label_path = full_path + ".txt" # datasets2/track/KJ-C1-08AM/crops\KJ-C1-08AM-frame_007489.txt
    label_path = label_path.replace("crops", "labels") # datasets2/track/KJ-C1-08AM/labels\KJ-C1-08AM-frame_007489.txt

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
    if "results" not in st.session_state:
        st.session_state.results = []
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
    if "bdel1_disabled" not in st.session_state:
        st.session_state.bdel1_disabled = False
    if "bdel2_disabled" not in st.session_state:
        st.session_state.bdel2_disabled = False
    # if "bsave_disabled" not in st.session_state:
    #     st.session_state.bsave_disabled = True
    # if "new_img_list1" not in st.session_state:
    #     st.session_state.new_img_list1 = []
    # if "new_img_list2" not in st.session_state:
    #     st.session_state.new_img_list2 = []

def start_comparison():
    """Callback for the Run button"""
    try:
        # Initialize lists
        crop_files1 = image_listdir(st.session_state.crop_dir1)
        crop_files2 = image_listdir(st.session_state.crop_dir2)
        st.session_state.image_list1 = filter_files(crop_files1)
        st.session_state.image_list2 = filter_files(crop_files2)

        st.session_state.is_running = True
    except FileNotFoundError:
        st.error("Invalid path!")
    except Exception as e:
        st.error("Something went wrong", e)

# def resume_comparison():
#     st.session_state.is_running = True
#     # Initialize lists
#     crop_files1 = image_listdir(st.session_state.crop_dir1)
#     crop_files2 = image_listdir(st.session_state.crop_dir2)
#     st.session_state.image_list1 = filter_files(crop_files1)
#     st.session_state.image_list2 = filter_files(crop_files2)
#     st.session_state.image_list1 = st.session_state.new_img_list1
#     st.session_state.image_list2 = st.session_state.new_img_list2

def next1():
    if len(st.session_state.image_list1) > 1:
        st.session_state.img1 += 1

def back1():
    if len(st.session_state.image_list1) > 1:
        st.session_state.img1 -= 1

def next2():
    if len(st.session_state.image_list2) > 1:
        st.session_state.img2 += 1

def back2():
    if len(st.session_state.image_list2) > 1:
        st.session_state.img2 -= 1

def del1():
    if len(st.session_state.image_list1) > 0:
        current_image = st.session_state.image_list1[st.session_state.img1]
        st.session_state.image_list1.remove(current_image)
        st.session_state.img1 = min(st.session_state.img1, len(st.session_state.image_list1) - 1) 

def del2():
    if len(st.session_state.image_list2) > 0:
        current_image = st.session_state.image_list2[st.session_state.img2]
        st.session_state.image_list2.remove(current_image)
        st.session_state.img2 = min(st.session_state.img2, len(st.session_state.image_list2) - 1) 

def match():
    st.session_state.results.append({
        "image1": os.path.splitext(st.session_state.image_list1[st.session_state.img1])[0].split("-")[-1],
        "image2": os.path.splitext(st.session_state.image_list2[st.session_state.img2])[0].split("-")[-1],
    })
    
    if len(st.session_state.image_list1) > 0:
        current_image = st.session_state.image_list1[st.session_state.img1]
        st.session_state.image_list1.remove(current_image)
        st.session_state.img1 = min(st.session_state.img1, len(st.session_state.image_list1) - 1) 

    if len(st.session_state.image_list2) > 0:
        current_image = st.session_state.image_list2[st.session_state.img2]
        st.session_state.image_list2.remove(current_image)
        st.session_state.img2 = min(st.session_state.img2, len(st.session_state.image_list2) - 1) 

def save():
    try:
        save_results(st.session_state.save_result_path, st.session_state.results, mode='a')
        st.success(f"Result file successfully saved to {st.session_state.save_result_path}")
    except (FileNotFoundError, PermissionError):
        st.error("Save path is invalid!")
    except Exception:
        st.error("Please check your save path!")

# def resume():
#     resume_comparison()
#     st.session_state.resume_img_list1, st.session_state.resume_img_list2 = read_last_imgs(st.session_state.resume_file_path)
#     # st.write(st.session_state.resume_img_list1, st.session_state.resume_img_list2)

#     st.session_state.current_img_list1 = [os.path.splitext(image1)[0].split("-")[-1] for image1 in st.session_state.image_list1]
#     st.session_state.current_img_list2 = [os.path.splitext(image2)[0].split("-")[-1] for image2 in st.session_state.image_list2]

#     # st.write(st.session_state.current_img_list1)
#     # st.write(st.session_state.current_img_list2)

#     # st.session_state.new_img_list1 = set(st.session_state.resume_img_list1) ^ set(st.session_state.current_img_list1)
#     st.session_state.new_img_list1 = [item for item in st.session_state.resume_img_list1 if item not in st.session_state.current_img_list1] + [item for item in st.session_state.current_img_list1 if item not in st.session_state.resume_img_list1]
#     st.session_state.new_img_list2 = [item for item in st.session_state.resume_img_list2 if item not in st.session_state.current_img_list2] + [item for item in st.session_state.current_img_list2 if item not in st.session_state.resume_img_list2]
    
#     # START HERE

#     # FileNotFoundError: [Errno 2] No such file or directory: '000000'
#     # CHANGE IT BACK TO path
    
#     st.write(st.session_state.image_list1)
#     st.write(st.session_state.image_list2)
    
#     # st.write(st.session_state.new_img_list1)
#     start_comparison()

#     st.write(st.session_state.image_list1)
#     st.write(st.session_state.image_list2)
#     # save_last_imgs(resume_path, done_img1, done_img2)