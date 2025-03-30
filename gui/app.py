from pathlib import Path
import streamlit as st
import pandas as pd
import os
import albumentations as A

import settings
import helper
import yolo_crop
import comparison
import rename
import datasplit
import reid
import visualization
import prepare_VeRi
import database
import augment

# initialize path and directories
model_path = Path(settings.YOLO_MODEL_FILEPATH)

video_dir = Path(settings.VIDEO_DIR)
video_path = Path(settings.VIDEO_FILEPATH)

output_dir = Path(settings.OUTPUT_DIR)
crops_dir = Path(settings.CROPS_DIR)
datasets_dir = Path(settings.DATASETS_DIR)

db_path = Path(settings.DATABASE_FILEPATH)

# remove the video files after task completed to save storage
helper.cleanup(video_dir)

st.set_page_config(
    page_title="Vehicle Re-ID",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.header("Tasks Selection")
with st.sidebar.container(border=True):
    if "task_option" not in st.session_state:
        st.session_state.task_option = ["Data Preparation", "Image Comparison", "Data Augmentation", "Dataset Split", "Model Training", "Model Testing", "Visualization", "Database Management"]

    task_type = st.radio("Tasks Selection", options=st.session_state.task_option, label_visibility="collapsed")

########################

st.sidebar.header("Task Configuration")
if task_type == "Data Preparation":
    with st.sidebar.container(border=True):
        uploaded_video = st.file_uploader("Select video file", type=['mp4'])

        confidence = float(st.slider("Model Confidence Level", 25, 100, 80)) / 100
        batchsize = st.number_input("Batch Size", min_value=10, value=100, step=10, help="Batch size in number of frames")

        if "location" not in st.session_state:
            st.session_state.location = None
        if "camera_id" not in st.session_state:
            st.session_state.camera_id = None
        if "time" not in st.session_state:
            st.session_state.time = None

        st.session_state.location = st.selectbox("Location", options=["KJ", "SJ", "Custom"])
        if st.session_state.location == "Custom":
            st.session_state.location = st.text_input("Custom Location", placeholder="Max characters: 6")
            # Validate input length
            if st.session_state.location and len(st.session_state.location) > 6:
                st.warning("Maximum 6 characters only!")
                st.session_state.location = st.session_state.location[:6]  # automatically truncate the input

        st.session_state.camera_id = st.selectbox("Camera ID", options=["C1", "C2", "C3", "C4", "Custom"])
        if st.session_state.camera_id == "Custom":
            st.session_state.camera_id = st.text_input("Custom Camera ID", placeholder="Max characters: 6")
            if st.session_state.camera_id and len(st.session_state.camera_id) > 6:
                st.warning("Maximum 6 characters only!")
                st.session_state.camera_id = st.session_state.camera_id[:6]

        st.session_state.time = st.selectbox("Time", options=["0800", "1100", "1500", "2000", "Custom"])
        if st.session_state.time == "Custom":
            st.session_state.time = st.text_input("Custom Time", placeholder="Max characters: 6")
            if st.session_state.time and len(st.session_state.time) > 6:
                st.warning("Maximum 6 characters only!")
                st.session_state.time = st.session_state.time[:6]

        prefix = f"{st.session_state.location}-{st.session_state.camera_id}-{st.session_state.time}"
        
    if (uploaded_video is not None) and (all(x != "" for x in [st.session_state.location, st.session_state.camera_id, st.session_state.time])):
        st.header("Video Preview")
        with st.container(border=True):
            col1, col2, col3 = st.columns([0.3, 0.5, 0.3])
            with col2: # center
                st.video(uploaded_video)

        video_byte = uploaded_video.getvalue()
        helper.save_file(video_byte, video_dir)

        run_button = st.sidebar.button("Run", type="primary", use_container_width=True)
    else:
        run_button = st.sidebar.button("Run", type="primary", use_container_width=True, disabled=True)
    
    if run_button:
        with st.spinner("Running..."):
            database.create_table("video")
            database.insert_data("video")

            new_output_dir = helper.create_subfolders(output_dir, "output")
            yolo_crop.track(model_path, video_path, new_output_dir, batchsize=batchsize, conf=confidence, save_frames=True, save_txt=True, prefix=prefix)
            yolo_crop.save_crop(new_output_dir)

        st.success(f'Done! Output files saved to "{new_output_dir}"')
        
    cancel_button = st.sidebar.button("Cancel", use_container_width=True)
    if cancel_button:
        st.toast("Operation cancelled.")
        st.stop()

########################

if task_type == "Image Comparison":
    comparison.initialize_session_state()

    with st.sidebar.container(border=True):
        st.session_state.crop_dir1 = st.text_input("First Crop Directory Path", value=os.path.join(output_dir, "output/crops"), key="crop_dir1_input")
        st.session_state.crop_dir2 = st.text_input("Second Crop Directory Path", value=os.path.join(output_dir, "output2/crops"), key="crop_dir2_input")

        save_both = st.checkbox("Save both results")
        show_del_undo = st.checkbox('Show "Delete" & "Undo"')

        if os.path.isfile(db_path):
            st.button("Save progress", use_container_width=True, on_click=comparison.save)
            st.button("Resume checkpoint", use_container_width=True, on_click=comparison.resume)
            st.button("Delete checkpoint", type="primary", use_container_width=True, on_click=comparison.reset)
        
    paths = [st.session_state.crop_dir1, st.session_state.crop_dir2]
    path_not_exists = any(not os.path.exists(path) for path in paths)
    st.sidebar.button("Run", type="primary", use_container_width=True, disabled=path_not_exists, on_click=comparison.start_comparison)

    # error check
    if not os.path.isdir(st.session_state.crop_dir1):
        st.error("First crop directory path is not found!")
    if not os.path.isdir(st.session_state.crop_dir2):
        st.error("Second crop directory path is not found!")

    save_button = st.sidebar.button("Save results", type="primary", use_container_width=True, disabled=not os.path.isfile(db_path))
    if save_button:
        with st.spinner("Running..."):
            if comparison.save():
                if save_both:
                    new_crop_dir1 = helper.create_subfolders(crops_dir, "crop")
                    new_crop_dir2 = helper.create_subfolders(crops_dir, "crop")
                    rename.rename_files(st.session_state.crop_dir1, new_crop_dir1, st.session_state.crop_dir2, new_crop_dir2)
                    st.success(f'Results successfully saved to "{new_crop_dir1}" and "{new_crop_dir2}".')
                else: 
                    # save only the second results
                    new_crop_dir2 = helper.create_subfolders(crops_dir, "crop")
                    rename.rename_files(st.session_state.crop_dir2, new_crop_dir2)
                    st.success(f'Results successfully saved to "{new_crop_dir2}".')
    
    # Only show comparison interface if running
    if st.session_state.is_running:
        if len(st.session_state.image_list1) == 0 or len(st.session_state.image_list2) == 0:
            st.success("Comparison completed! Remember to save your results :)")
        else:
            # prevent list index out of range error
            st.session_state.img1 = min(st.session_state.img1, len(st.session_state.image_list1) - 1)
            st.session_state.img2 = min(st.session_state.img2, len(st.session_state.image_list2) - 1)

            img1_path, label1_path = comparison.get_image_label_path(st.session_state.image_list1[st.session_state.img1])
            img2_path, label2_path = comparison.get_image_label_path(st.session_state.image_list2[st.session_state.img2])

            # Update button states
            st.session_state.bnext1_disabled = st.session_state.img1 == len(st.session_state.image_list1) - 1
            st.session_state.bback1_disabled = st.session_state.img1 <= 0
            st.session_state.bnext2_disabled = st.session_state.img2 == len(st.session_state.image_list2) - 1
            st.session_state.bback2_disabled = st.session_state.img2 <= 0
            st.session_state.bmatch_disabled = (len(st.session_state.image_list1) == 0) or (len(st.session_state.image_list2) == 0)
            st.session_state.bundo_match_disabled = (len(st.session_state.undo_stack1) == 0) or (len(st.session_state.undo_stack2) == 0)
            st.session_state.bdel1_disabled = len(st.session_state.image_list1) == 0
            st.session_state.bdel2_disabled = len(st.session_state.image_list2) == 0
            st.session_state.bundo_del1_disabled = len(st.session_state.undo_stack1) == 0
            st.session_state.bundo_del2_disabled = len(st.session_state.undo_stack2) == 0
                
            # Navigation buttons
            bcol1, bcol2, bcol3, bcol4 = st.columns(4, vertical_alignment="bottom")
            bcol1.button("Previous", use_container_width=True, key="bback1", disabled=st.session_state.bback1_disabled, on_click=comparison.back1)
            bcol2.button("Next", use_container_width=True, key="bnext1", disabled=st.session_state.bnext1_disabled, on_click=comparison.next1)
            bcol3.button("Previous", use_container_width=True, key="bback2", disabled=st.session_state.bback2_disabled, on_click=comparison.back2)
            bcol4.button("Next", use_container_width=True, key="bnext2", disabled=st.session_state.bnext2_disabled, on_click=comparison.next2)

            # Action buttons
            st.button("Match found", type="primary", use_container_width=True, disabled=st.session_state.bmatch_disabled, on_click=comparison.match)
            st.button("Undo match found", use_container_width=True, disabled=st.session_state.bundo_match_disabled, on_click=comparison.undo_match)

            # Display images
            with st.container(border=True):
                comparison.compare_images(st.session_state.image_list1[st.session_state.img1], st.session_state.image_list2[st.session_state.img2], img1_path, img2_path, label1_path, label2_path)
            
            if show_del_undo:
                # Delete buttons
                bcol5, bcol6 = st.columns(2, vertical_alignment="bottom")
                bcol5.button("Delete", type="primary", use_container_width=True, key="bdel1", disabled=st.session_state.bdel1_disabled, on_click=comparison.del1)
                bcol6.button("Delete", type="primary", use_container_width=True, key="bdel2", disabled=st.session_state.bdel2_disabled, on_click=comparison.del2)

                # Undo buttons 
                bcol7, bcol8 = st.columns(2, vertical_alignment="bottom")
                bcol7.button("Undo", use_container_width=True, key="bundo_del1", disabled=st.session_state.bundo_del1_disabled, on_click=comparison.undo_del1)
                bcol8.button("Undo", use_container_width=True, key="bundo_del2", disabled=st.session_state.bundo_del2_disabled, on_click=comparison.undo_del2)

########################

if task_type == "Data Augmentation":
    with st.sidebar.container(border=True):
        transform_list = []

        image_path = st.text_input("Images Directory Path", value=os.path.join(crops_dir, "crop"), key="augment_images_path")
        output_path = st.text_input("Output Directory Path", value=os.path.join(crops_dir, "_crop"), key="augment_output_path")

        st.write("Apply augmentation techniques:")
        horflip = st.checkbox("HorizontalFlip", help="Flip the input horizontally around the y-axis.")
        if horflip:
            horflip_p = st.slider("HorizontalFlip Probability", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
            transform_list.append(A.HorizontalFlip(p=horflip_p))

        verflip = st.checkbox("VerticalFlip", help="Flip the input vertically around the x-axis.")
        if verflip:
            verflip_p = st.slider("VerticalFlip Probability", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
            transform_list.append(A.VerticalFlip(p=verflip_p))

        rotate = st.checkbox("RandomRotate90", help="Randomly rotate the input by 90 degrees zero or more times.")
        if rotate:
            rotate_p = st.slider("RandomRotate90 Probability", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
            transform_list.append(A.RandomRotate90(p=rotate_p))

        erasing = st.checkbox("Erasing", help="Randomly erases rectangular regions in an image, following the Random Erasing Data Augmentation technique.")
        if erasing:
            erasing_p = st.slider("Erasing Probability", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
            transform_list.append(A.Erasing(p=erasing_p))

        colorjitter = st.checkbox("ColorJitter", help="Randomly changes the brightness, contrast, saturation, and hue of an image.")
        if colorjitter:
            colorjitter_p = st.slider("ColorJitter Probability", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
            transform_list.append(A.ColorJitter(p=colorjitter_p))

        sharpen = st.checkbox("Sharpen", help="Sharpen the input image using either kernel-based or Gaussian interpolation method.")
        if sharpen:
            sharpen_p = st.slider("Sharpen Probability", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
            transform_list.append(A.Sharpen(p=sharpen_p))

        unsharpmask = st.checkbox("UnsharpMask", help="Sharpen the input image using Unsharp Masking processing and overlays the result with the original image.")
        if unsharpmask:
            unsharpmask_p = st.slider("UnsharpMask Probability", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
            transform_list.append(A.UnsharpMask(p=unsharpmask_p))

        clahe = st.checkbox("CLAHE", help="Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to the input image.")
        if clahe:
            clahe_p = st.slider("CLAHE Probability", min_value=0.0, max_value=1.0, step=0.1, value=0.5)
            transform_list.append(A.CLAHE(p=clahe_p))

        seed = st.number_input(label="Set random seed", value=137)

    paths = [image_path]
    path_not_exists = any(not os.path.exists(path) for path in paths)
    run_button = st.sidebar.button("Run", type="primary", use_container_width=True, disabled=path_not_exists)

    # error check
    if not os.path.isdir(image_path):
        st.error("Images directory path is not found!")

    if run_button:
        with st.spinner("Running..."):
            augment.augment(transform_list, seed, image_path, output_path)
        st.success(f'Images successfully augmented in "{image_path}"')

########################

if task_type == "Dataset Split":
    with st.sidebar.container(border=True):
        crop_dir = st.text_input("Crop Directory Path", value=crops_dir, key="crop_dir_input")
        dataset_name = st.text_input("Dataset Name", value="reid", key="dataset_name_input")
        split_ratio = st.slider("Split Ratio", help="Train & Validation Split Ratio", min_value=0.0, max_value=1.0, step=0.05, value=0.75)
        seed = st.number_input(label="Set random seed", value=42)

        custom_dataset = st.checkbox("Custom Dataset")
        if custom_dataset:
            dataset_selection = st.radio("Custom Dataset", options=["VeRi-776"], label_visibility="collapsed")
            if dataset_selection == "VeRi-776":
                VeRi_dir = st.text_input("Dataset Directory Path", value=os.path.join(datasets_dir, "VeRi"))
                train_path = os.path.join(VeRi_dir, "name_train.txt")
                train_csv_path = os.path.join(VeRi_dir, "train.csv")

                test_path = os.path.join(VeRi_dir, "name_test.txt")
                test_csv_path = os.path.join(VeRi_dir, "gallery.csv")

                query_path = os.path.join(VeRi_dir, "name_query.txt")
                query_csv_path = os.path.join(VeRi_dir, "query.csv")

    paths = [crop_dir]
    path_not_exists = any(not os.path.exists(path) for path in paths)
    run_button = st.sidebar.button("Run", type="primary", use_container_width=True, disabled=path_not_exists)

    # error check
    if not os.path.isdir(crop_dir):
        st.error("Crop directory path is not found!")

    if run_button:
        with st.spinner("Running..."):
            if custom_dataset:
                if dataset_selection == "VeRi-776":
                    prepare_VeRi.txt_to_csv(train_path, train_csv_path, "image_train")
                    datasplit.train_val_split(VeRi_dir, split_ratio)
                    prepare_VeRi.txt_to_csv(test_path, test_csv_path, "image_test")
                    prepare_VeRi.txt_to_csv(query_path, query_csv_path, "image_query")
                    st.success(f"Datasets successfully created at {VeRi_dir}")
            else:
                dataset_dir = helper.create_subfolders(datasets_dir, dataset_name)
                datasplit.datasplit(crop_dir, dataset_dir, split_ratio, seed)
                st.success(f"Datasets successfully created at {dataset_dir}")

########################

if task_type == "Model Training":
    with st.sidebar.container(border=True):
        data_dir = st.text_input("Dataset Directory Path", value=os.path.join(datasets_dir, "reid"), help="Path to the dataset root directory path")
        train_csv_path = os.path.join(data_dir, "train.csv")
        val_csv_path = os.path.join(data_dir, "val.csv")

        name = st.text_input("Output Model Name", value="resnet50", help="Define custom output model name")

        batchsize = st.number_input("Batch Size", min_value=8, value=32, step=8)
        total_epoch = st.number_input("Total Epoch", min_value=1, value=60)

        advanced = st.toggle("Advanced Settings")

        if advanced:
            model = st.selectbox("Model", options=['resnet', 'resnet_ibn', 'densenet', 'swin', 'NAS', 
                                                        'hr', 'efficientnet'], index=1)
            if model == "efficientnet":
                subtype = st.slider("Model Subtype", min_value=0, max_value=7)
                model_subtype = "b" + str(subtype)

            warm_epoch = st.number_input("Warm Epoch", min_value=0, value=0)
            save_freq = st.number_input("Save Frequency", min_value=1, value=5)
            num_workers = st.number_input("Number of Workers", min_value=1, value=2)
            lr = st.number_input("Learning Rate", min_value=0.01, value=0.05)
            erasing_p = st.slider("Random Erasing Probability", min_value=0.0, max_value=1.0, step=0.05, value=0.50)
            fp16 = st.checkbox("fp16", help="Use mixed precision training. This will occupy less memory in the forward pass, and will speed up training in some architectures (Nvidia A100, V100, etc.)")
            cosine = st.checkbox("Cosine", help="Use cosine learning rate")
            color_jitter = st.checkbox("Color jitter", help="Use color jitter in training")
            st.subheader("Loss Functions")
            triplet = st.checkbox("Triplet Loss", help="Use Triplet Loss function")
            contrast = st.checkbox("Contrastive Loss", help="Use Contrastive Loss function")
            sphere = st.checkbox("Sphere Loss", help="Use Sphere Loss function")
            circle = st.checkbox("Circle Loss", help="Use Circle Loss function")
        else:
            model="resnet_ibn"
            model_subtype="default"
            warm_epoch=0
            save_freq=5
            num_workers=2
            lr=0.05
            erasing_p=0.5
            fp16=False
            cosine=False
            color_jitter=False
            triplet=False
            contrast=False
            sphere=False
            circle=False

    paths = [data_dir, train_csv_path, val_csv_path]
    path_not_exists = any(not os.path.exists(path) for path in paths)
    run_button = st.sidebar.button("Run", type="primary", use_container_width=True, disabled=path_not_exists)
    cancel_button = st.sidebar.button("Cancel", use_container_width=True, on_click=reid.stop_process)

    # error check
    if not os.path.isdir(data_dir):
        st.error("Dataset directory path is not found!")
    if not os.path.isfile(train_csv_path):
        st.error("train.csv file not found in dataset directory path!")
    if not os.path.isfile(val_csv_path):
        st.error("val.csv file not found in dataset directory path!")
    
    if run_button:
        with st.spinner("Running..."):
            with st.container(border=True):
                reid.initialize_session_state()
                reid.train("train.py", data_dir=data_dir, train_csv_path=train_csv_path, 
                            val_csv_path=val_csv_path, name=name, batchsize=batchsize, 
                            total_epoch=total_epoch, model=model, model_subtype=model_subtype,
                            warm_epoch=warm_epoch, save_freq=save_freq, num_workers=num_workers,
                            lr=lr, erasing_p=erasing_p, fp16=fp16, cosine=cosine,
                            color_jitter=color_jitter, triplet=triplet, contrast=contrast,
                            sphere=sphere, circle=circle)
        st.success("Done!")

########################

if task_type == "Model Testing":
    with st.sidebar.container(border=True):
        data_dir = st.text_input("Dataset Directory Path", value=os.path.join(datasets_dir, "reid"), help="Path to the dataset root directory path")
        query_csv_path = os.path.join(data_dir, "query.csv")
        gallery_csv_path = os.path.join(data_dir, "gallery.csv")
        
        model_dir = st.text_input("Model Directory Path:", value="model/resnet50", help="Path to the model root directory path")
        model_name = os.path.basename(model_dir)
        model_opts = os.path.join(model_dir, "opts.yaml")
        checkpoint = st.text_input("Model pth Filename", value="net_59.pth", help="Model pth file")
        checkpoint = os.path.join(model_dir, checkpoint)

        batchsize = st.number_input("Batch Size", min_value=8, value=32, step=8)

        eval_gpu = st.checkbox("GPU", value=True, help="Run evaluation on GPU. This may need a high amount of GPU memory.")

    paths = [data_dir, query_csv_path, gallery_csv_path, model_dir, model_opts, checkpoint]
    path_not_exists = any(not os.path.exists(path) for path in paths)
    run_button = st.sidebar.button("Run", type="primary", use_container_width=True, disabled=path_not_exists)

    # error check
    if not os.path.isdir(data_dir):
        st.error("Dataset directory path is not found!")
    if not os.path.isfile(query_csv_path):
        st.error("query.csv file not found in dataset directory path!")
    if not os.path.isfile(gallery_csv_path):
        st.error("gallery.csv file not found in dataset directory path!")

    if not os.path.isdir(model_dir):
        st.error("Model directory path is not found!")
    if not os.path.isfile(model_opts):
        st.error("opts.yaml file not found in model directory path!")
    if not os.path.isfile(checkpoint):
        st.error("Model pth file not found in model directory path!")
    
    if run_button:
        with st.spinner("Running..."):
            with st.container(border=True):
                test_result_path = os.path.join(model_dir, "train.jpg")
                st.image(test_result_path)
            with st.container(border=True):
                reid.initialize_session_state()
                reid.test("test.py", data_dir=data_dir, query_csv_path=query_csv_path,
                           gallery_csv_path=gallery_csv_path, model_opts=model_opts,
                           checkpoint=checkpoint, name=model_name, batchsize=batchsize, eval_gpu=eval_gpu)
        st.success("Done!")

########################

if task_type == "Visualization":
    with st.sidebar.container(border=True):
        data_dir = st.text_input("Dataset Directory Path", value=os.path.join(datasets_dir, "reid"), help="Path to the dataset root directory path")
        query_csv_path = os.path.join(data_dir, "query.csv")
        gallery_csv_path = os.path.join(data_dir, "gallery.csv")
        
        model_dir = st.text_input("Model Directory Path:", value="model/resnet50", help="Path to the model root directory path")
        model_opts = os.path.join(model_dir, "opts.yaml")
        checkpoint = st.text_input("Model pth Filename", value="net_59.pth", help="Model pth file")
        checkpoint = os.path.join(model_dir, checkpoint)

        advanced = st.toggle("Advanced Settings")

        if advanced:
            batchsize = st.number_input("Batch Size", min_value=8, value=64, step=8)
            input_size = st.number_input("Image Input Size", min_value=128, value=224, help="Image input size for the model")
            num_images = st.number_input("Number of Gallery Images", min_value=2, value=29, help="Number of gallery images to show")
            imgs_per_row = st.number_input("Images Per Row", min_value=2, value=6)

            use_saved_mat = st.checkbox("Use Cache", value=True, help="Use precomputed features from a previous test.py run: pytorch_result.mat")
        else:
            batchsize = 64
            input_size = 224
            num_images = 29
            imgs_per_row = 6
            use_saved_mat = False
        
    # error check
    if not os.path.isdir(data_dir):
        st.error("Dataset directory path is not found!")
    if not os.path.isfile(query_csv_path):
        st.error("query.csv file not found in dataset directory path!")
    if not os.path.isfile(gallery_csv_path):
        st.error("gallery.csv file not found in dataset directory path!")

    if not os.path.isdir(model_dir):
        st.error("Model directory path is not found!")
    if not os.path.isfile(model_opts):
        st.error("opts.yaml file not found in model directory path!")
    if not os.path.isfile(checkpoint):
        st.error("Model pth file not found in model directory path!")

    paths = [data_dir, query_csv_path, gallery_csv_path, model_dir, model_opts, checkpoint]
    path_not_exists = any(not os.path.exists(path) for path in paths)
    if not path_not_exists: # if all the files exist
        queries_len = len(pd.read_csv(query_csv_path)) - 1
        curr_idx = st.slider("Select Query Image Index", min_value=0, max_value=queries_len, value=0)

        with st.spinner("Running..."):
            with st.container(border=True):
                fig = visualization.visualize(data_dir=data_dir, query_csv_path=query_csv_path, gallery_csv_path=gallery_csv_path, model_opts=model_opts,
                                                checkpoint=checkpoint, batchsize=batchsize, input_size=input_size, num_images=num_images, 
                                                imgs_per_row=imgs_per_row, use_saved_mat=use_saved_mat, curr_idx=curr_idx)
                st.pyplot(fig)

########################

if task_type == "Database Management":
    with st.sidebar.container(border=True):
        if "db_table" not in st.session_state:
            database.init_db_connection() # create database file if not exists
            st.session_state.db_table = database.get_table_names()

        if not st.session_state.db_table: # check if list is empty
            st.warning("Database is currently empty.")
        
        table_selected = st.radio("Edit Database Table", options=sorted(st.session_state.db_table), label_visibility="visible")

        st.button("Refresh database", use_container_width=True, on_click=database.refresh)
        st.button("Backup database", use_container_width=True, on_click=database.backup)
        st.button("Delete database", type="primary", use_container_width=True, on_click=database.dialog_delete_db)

    if table_selected:
        st.subheader(f'"{table_selected}" table')
        df = database.db_to_df(table_selected) # convert database to dataframe
        edited_df = st.data_editor(df, use_container_width=True, disabled=("id", "video_id", "image_id", "crop_id")) # show all rows
        database.df_to_db(edited_df, table_selected) # apply changes to database