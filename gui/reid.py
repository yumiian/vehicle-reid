import subprocess
import streamlit as st
import time
import os
from pathlib import Path
import collections
import gc
import torch

# Get the parent directory of the current file
SCRIPT_DIR = Path(__file__).parent.parent

def initialize_session_state():
    if "process" not in st.session_state:
        st.session_state.process = None
    if "process_stop" not in st.session_state:
        st.session_state.process_stop = False
    if "output_area" not in st.session_state:
        st.session_state.output_area = st.empty()
    if "display_lines" not in st.session_state:
        st.session_state.display_lines = collections.deque(maxlen=500)  # Limit number of lines for display
    if "last_update" not in st.session_state:
        st.session_state.last_update = time.time()

def stop_process():
    if st.session_state.process:
        st.session_state.process.terminate()  # Terminate the process safely
        st.session_state.display_lines.append("\nProcess terminated by user.")
        st.session_state.process_stop = True
        st.session_state.output_area.code("".join(st.session_state.display_lines), language="bash", height=700, wrap_lines=True) # display the output again
        gc.collect()
        torch.cuda.empty_cache()
        st.error("Operation cancelled.")

def reset_process():
    # Reset stop flag and clear previous output
    st.session_state.process_stop = False
    st.session_state.display_lines = collections.deque(maxlen=500)

def write_terminal_output(process_type, name):
    if st.session_state.process is None:
        st.error("Process is not found! Please try again.")
        return

    log_dir = os.path.join(SCRIPT_DIR, "model", name)
    os.makedirs(log_dir, exist_ok=True)

    # Find the next available filename
    i = 1
    while os.path.exists(os.path.join(log_dir, f"log_{process_type}_{i}.txt")):
        i += 1

    log_filename = f"log_{process_type}_{i}.txt"

    # create log file
    log_path = os.path.join(log_dir, log_filename)
    log_file = open(log_path, "w")

    process = st.session_state.process
    while True:
        if st.session_state.process_stop: # exit loop if user cancel
            break

        line = process.stdout.readline() # read next line from process
        if line:
            # write to log file
            log_file.write(line)
            log_file.flush()

            st.session_state.display_lines.append(line) # Add to display buffer (limited size)

            # Only update display periodically to reduce rendering overhead
            current_time = time.time()
            update_interval = 0.5
            if current_time - st.session_state.last_update > update_interval:
                st.session_state.output_area.code("".join(st.session_state.display_lines), language="bash", height=700, wrap_lines=True)
                st.session_state.last_update = current_time
        else:
            # If no new line is available, check if the process has finished
            if process.poll() is not None:
                st.session_state.output_area.code("".join(st.session_state.display_lines), language="bash", height=700, wrap_lines=True)
                break

        time.sleep(0.01) # slight delay for UI update

    # Ensure the process has terminated and clean up when the process finishes
    process.wait()
    if process.poll() is not None:
        st.session_state.process = None
    gc.collect()
    torch.cuda.empty_cache()
    log_file.close()

def train(file_path, data_dir, train_csv_path, val_csv_path, name="ft_ResNet50", batchsize=32, total_epoch=60, 
          model="resnet_ibn", model_subtype="default", warm_epoch=0, save_freq=5, num_workers=2, lr=0.05, samples_per_class=1,
          erasing_p=0.5, fp16=False, cosine=False, color_jitter=False, triplet=False, contrast=False,
          sphere=False, circle=False):
    command = [
        "python3", file_path,
        "--data_dir", data_dir,
        "--train_csv_path", train_csv_path,
        "--val_csv_path", val_csv_path,
        "--name", name,
        "--batchsize", str(batchsize),
        "--total_epoch", str(total_epoch),
        "--model", model,
        "--model_subtype", model_subtype,
        "--warm_epoch", str(warm_epoch),
        "--save_freq", str(save_freq),
        "--num_workers", str(num_workers),
        "--lr", str(lr),
        "--samples_per_class", str(samples_per_class),
        "--erasing_p", str(erasing_p),
    ]

    if fp16: command.append("--fp16")
    if cosine: command.append("--cosine")
    if color_jitter: command.append("--color_jitter")
    if triplet: command.append("--triplet")
    if contrast: command.append("--contrast")
    if sphere: command.append("--sphere")
    if circle: command.append("--circle")

    reset_process()

    st.session_state.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    write_terminal_output("train", name)

def test(file_path, data_dir, query_csv_path, gallery_csv_path, model_opts, checkpoint, name, batchsize=32, eval_gpu=False):
    command = [
        "python3", file_path,
        "--data_dir", data_dir,
        "--query_csv_path", query_csv_path,
        "--gallery_csv_path", gallery_csv_path,
        "--model_opts", model_opts,
        "--batchsize", str(batchsize),
        "--checkpoint", checkpoint,
    ]

    if eval_gpu: command.append("--eval_gpu")

    reset_process()

    st.session_state.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    write_terminal_output("test", name)