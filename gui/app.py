from pathlib import Path
import streamlit as st
import os

import settings
import helper
import yolo_crop
import comparison
import rename
import datasplit
import reid

st.set_page_config(
    page_title="Vehicle Reidentification",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.sidebar.title("Vehicle Reidentification")

#######################

model_path = Path(settings.YOLO_MODEL)
video_dir = Path(settings.VIDEO_DIR)
video_path = Path(settings.VIDEO_FILE)
save_path = Path(settings.OUTPUT_DIR)
new_result_path = Path(settings.NEW_RESULT_DIR)
datasets_path = Path(settings.DATASETS_DIR)

########################

helper.cleanup(video_dir)

########################

st.sidebar.header("Model Config")

task_type = st.sidebar.radio("Select Task", ["Create Crops", "Compare Images", "Batch Rename", 
                                             "Dataset Split", "Model Training", "Model Testing"])

########################

if task_type == "Create Crops":
    confidence = float(st.sidebar.slider("Select Model Confidence", 25, 100, 70)) / 100

    st.sidebar.header("Video Config")

    uploaded_video = st.sidebar.file_uploader("Upload a video", type=['mp4'])
    location = st.sidebar.text_input("Location", placeholder="eg. KJ, SJ")
    camera = st.sidebar.text_input("Camera ID", placeholder="eg. C1, C2, C3, C4")
    time = st.sidebar.text_input("Time", placeholder="eg. 8AM, 3PM, 8PM")
    prefix = f"{location}-{camera}-{time}"
    
    if (uploaded_video is not None) and (all(x is not "" for x in [location, camera, time])):
        # st.video(uploaded_video)

        video_byte = uploaded_video.getvalue()
        helper.save_file(video_byte, video_dir)

        run_button = st.sidebar.button("Run", type="primary", use_container_width=True)
    else:
        run_button = st.sidebar.button("Run", type="primary", use_container_width=True, disabled=True)
    
    if run_button:
        with st.spinner("Running..."):
            new_save_path = helper.create_subfolders(save_path, "result")
            yolo_crop.track(model_path, video_path, new_save_path, conf=confidence, save_frames=True, save_txt=True, prefix=prefix)
            yolo_crop.save_crop(new_save_path)
        st.success("Done!")
        
    cancel_button = st.sidebar.button("Cancel", use_container_width=True)
    if cancel_button:
        st.toast("Operation cancelled.")
        st.stop()

########################

if task_type == "Compare Images":
    comparison.initialize_session_state()
    # Sidebar inputs
    with st.sidebar:
        st.session_state.crop_dir1 = st.text_input("Input first crop folder path", value="gui/results/result/crops", key="crop_dir1_input")
        st.session_state.crop_dir2 = st.text_input("Input second crop folder path", value="gui/results/result2/crops", key="crop_dir2_input")

        save_result = st.checkbox("Save results", value=False)
        if save_result:
            st.session_state.save_result_path = st.text_input("Save path here", key="save_path", placeholder="result.txt")
            st.button("Save result", use_container_width=True, on_click=comparison.save)
        
        # resume = st.checkbox("Resume", value=False)
        # if resume:
        #     st.session_state.resume_file_path = st.text_input("Resume path here", key="resume_path")
        #     st.button("Resume", type="primary", use_container_width=True, on_click=verification.resume)
        
        st.button("Run", type="primary", use_container_width=True, on_click=comparison.start_comparison)
    
    # Only show comparison interface if running
    if st.session_state.is_running:
        if (len(st.session_state.image_list1) > 0) and (len(st.session_state.image_list2) > 0):
            img1_path, label1_path = comparison.get_image_label_path(st.session_state.image_list1[st.session_state.img1])
            img2_path, label2_path = comparison.get_image_label_path(st.session_state.image_list2[st.session_state.img2])
            
            # Display images
            with st.container(border=True):
                comparison.compare_images(st.session_state.image_list1[st.session_state.img1], st.session_state.image_list2[st.session_state.img2], img1_path, img2_path, label1_path, label2_path)

            # Update button states
            st.session_state.bnext1_disabled = st.session_state.img1 == len(st.session_state.image_list1) - 1
            st.session_state.bback1_disabled = st.session_state.img1 <= 0
            st.session_state.bnext2_disabled = st.session_state.img2 == len(st.session_state.image_list2) - 1
            st.session_state.bback2_disabled = st.session_state.img2 <= 0
            st.session_state.bmatch_disabled = len(st.session_state.image_list1) == 0
                
            # Navigation buttons
            bcol1, bcol2, bcol3, bcol4 = st.columns(4, vertical_alignment="bottom")
            bcol1.button("Previous", use_container_width=True, key="bback1", disabled=st.session_state.bback1_disabled, on_click=comparison.back1)
            bcol2.button("Next", use_container_width=True, key="bnext1", disabled=st.session_state.bnext1_disabled, on_click=comparison.next1)
            bcol3.button("Previous", use_container_width=True, key="bback2", disabled=st.session_state.bback2_disabled, on_click=comparison.back2)
            bcol4.button("Next", use_container_width=True, key="bnext2", disabled=st.session_state.bnext2_disabled, on_click=comparison.next2)
            
            # Action buttons
            st.button("Match found", type="primary", use_container_width=True, disabled=st.session_state.bmatch_disabled, on_click=comparison.match)
        else:
            st.write("Done comparison! Please save your results :)")

########################

if task_type == "Batch Rename":
    with st.sidebar:
        crop_dir1 = st.text_input("Input first crop folder path", value="gui/results/result/crops", key="crop_dir1_input")
        crop_dir2 = st.text_input("Input second crop folder path", value="gui/results/result2/crops", key="crop_dir2_input")
        save_result_path = st.text_input("Save path here", key="save_path")
        run_button = st.button("Run", type="primary", use_container_width=True)

        if run_button:
            with st.spinner("Running..."):
                new_crop_dir1 = helper.create_subfolders(new_result_path, "crop1")
                new_crop_dir2 = helper.create_subfolders(new_result_path, "crop2")
                rename.rename_files(crop_dir1, crop_dir2, new_crop_dir1, new_crop_dir2, save_result_path)
            st.success("Done!")

########################

if task_type == "Dataset Split":
    crop_dir1 = st.sidebar.text_input("Input first crop folder path", value="gui/new_results/crop1", key="crop_dir1_input")
    crop_dir2 = st.sidebar.text_input("Input second crop folder path", value="gui/new_results/crop2", key="crop_dir2_input")
    # output_path = st.sidebar.text_input("Save path here", key="save_path")
    run_button = st.sidebar.button("Run", type="primary", use_container_width=True)

    if run_button:
        with st.spinner("Running..."):
            dataset_dir = helper.create_subfolders(datasets_path, "reid")
            datasplit.datasplit(crop_dir1, crop_dir2, dataset_dir)
        st.success("Done!")

########################

if task_type == "Model Training":
    data_dir = st.sidebar.text_input("Dataset directory:", value="gui/datasets/reid", help="Path to the dataset root directory")
    train_csv_path = st.sidebar.text_input("Train csv file path:", value="gui/datasets/reid/train.csv", help="train.csv file path")
    val_csv_path = st.sidebar.text_input("Val csv file path:", value="gui/datasets/reid/val.csv", help="val.csv file path")
    name = st.sidebar.text_input("Output model name:", value="resnet50", help="Define custom output model name")

    batchsize = st.sidebar.number_input("Batch size", min_value=8, value=32, step=8)
    total_epoch = st.sidebar.number_input("Total epoch", min_value=1, value=60)

    advanced = st.sidebar.toggle("Advanced Settings")

    if advanced:
        model = st.sidebar.selectbox("Model", options=['resnet', 'resnet_ibn', 'densenet', 'swin', 'NAS', 
                                                       'hr', 'efficientnet'], index=1)
        if model == "efficientnet":
            subtype = st.sidebar.slider("Model Subtype", min_value=0, max_value=7)
            model_subtype = "b" + str(subtype)

        warm_epoch = st.sidebar.number_input("Warm epoch", min_value=0, value=0)
        save_freq = st.sidebar.number_input("Save frequency", min_value=1, value=5)
        num_workers = st.sidebar.number_input("Number of workers", min_value=1, value=2)
        lr = st.sidebar.number_input("Learning rate", min_value=0.01, value=0.05)
        erasing_p = st.sidebar.slider("Random Erasing probability", min_value=0.0, max_value=1.0, step=0.05, value=0.50)
        fp16 = st.sidebar.checkbox("fp16", help="Use mixed precision training. This will occupy less memory in the forward pass, and will speed up training in some architectures (Nvidia A100, V100, etc.)")
        cosine = st.sidebar.checkbox("Cosine", help="Use cosine learning rate")
        color_jitter = st.sidebar.checkbox("Color jitter", help="Use color jitter in training")
        triplet = st.sidebar.checkbox("Triplet Loss", help="Use Triplet Loss function")
        contrast = st.sidebar.checkbox("Contrast Loss", help="Use Contrast Loss function")
        sphere = st.sidebar.checkbox("Sphere Loss", help="Use Sphere Loss function")
        circle = st.sidebar.checkbox("Circle Loss", help="Use Circle Loss function")
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

    run_button = st.sidebar.button("Run", type="primary", use_container_width=True)
    if run_button:
        with st.spinner("Running..."):
            with st.container(border=True):
                # st.write(os.getcwd())
                stdout, stderr = reid.train("train.py", data_dir=data_dir, train_csv_path=train_csv_path, 
                                            val_csv_path=val_csv_path, name=name, batchsize=batchsize, 
                                            total_epoch=total_epoch, model=model, model_subtype=model_subtype,
                                            warm_epoch=warm_epoch, save_freq=save_freq, num_workers=num_workers,
                                            lr=lr, erasing_p=erasing_p, fp16=fp16, cosine=cosine,
                                            color_jitter=color_jitter, triplet=triplet, contrast=contrast,
                                            sphere=sphere, circle=circle)
                st.text(stdout + stderr)
                # st.error(stderr)
                # for line in out:
                #     st.text(line.strip())

                # for line in err:
                #     st.error(line.strip())
        st.success("Done!")

########################

if task_type == "Model Testing":
    data_dir = st.sidebar.text_input("Dataset directory:", value="gui/datasets/reid", help="Path to the dataset root directory")
    query_csv_path = st.sidebar.text_input("Query csv file path:", value="gui/datasets/reid/query.csv", help="query.csv file path")
    gallery_csv_path = st.sidebar.text_input("Gallery csv file path:", value="gui/datasets/reid/gallery.csv", help="gallery.csv file path")
    model_opts = st.sidebar.text_input("Model saved options", value="model/resnet50/opts.yaml", help="Model yaml file")
    checkpoint = st.sidebar.text_input("Model checkpoint path", value="model/resnet50/net_59.pth", help="Model pth file")
    batchsize = st.sidebar.number_input("Batch size", min_value=8, value=32, step=8)

    eval_gpu = st.sidebar.checkbox("GPU", help="Run evaluation on gpu too. This may need a high amount of GPU memory.")

    run_button = st.sidebar.button("Run", type="primary", use_container_width=True)
    if run_button:
        with st.spinner("Running..."):
            with st.container(border=True):
                stdout, stderr = reid.test("test.py", data_dir=data_dir, query_csv_path=query_csv_path,
                                           gallery_csv_path=gallery_csv_path, model_opts=model_opts,
                                           checkpoint=checkpoint, batchsize=batchsize, eval_gpu=eval_gpu)
                st.text(stdout + stderr)
                # st.error(stderr)
        st.success("Done!")