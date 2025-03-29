import subprocess
import streamlit as st
import time

def initialize_session_state():
    if "process" not in st.session_state:
        st.session_state.process = None
    if "process_stop" not in st.session_state:
        st.session_state.process_stop = False
    if "output_text" not in st.session_state:
        st.session_state.output_text = ""
    if "output_area" not in st.session_state:
        st.session_state.output_area = st.empty()

def stop_process():
    if st.session_state.process:
        st.session_state.process.terminate()  # Terminate the process safely
        st.session_state.output_text += "\nProcess terminated by user."
        st.session_state.process_stop = True
        st.session_state.output_area.code(st.session_state.output_text, language="bash", height=700, wrap_lines=True) # display the output again
        st.error("Operation cancelled.")

def write_terminal_output():
    if st.session_state.process is None:
        st.error("Process is not found! Please try again.")
        return
    
    process = st.session_state.process

    while True:
        if st.session_state.process_stop: # exit loop if user cancel
            break

        line = process.stdout.readline() # read next line from process
        if line:
            st.session_state.output_text += line
            st.session_state.output_area.code(st.session_state.output_text, language="bash", height=700, wrap_lines=True)
        else:
            # If no new line is available, check if the process has finished
            if process.poll() is not None:
                break

        time.sleep(0.01) # Slight delay to allow the UI to update

    # Ensure the process has terminated and clean up when the process finishes
    process.wait()
    if process.poll() is not None:
        st.session_state.process = None

def train(file_path, data_dir, train_csv_path, val_csv_path, name="ft_ResNet50", batchsize=32, total_epoch=60, 
          model="resnet_ibn", model_subtype="default", warm_epoch=0, save_freq=5, num_workers=2, lr=0.05,
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
        "--erasing_p", str(erasing_p),
    ]

    if fp16: command.append("--fp16")
    if cosine: command.append("--cosine")
    if color_jitter: command.append("--color_jitter")
    if triplet: command.append("--triplet")
    if contrast: command.append("--contrast")
    if sphere: command.append("--sphere")
    if circle: command.append("--circle")

    # Reset stop flag and clear previous output
    st.session_state.process_stop = False
    st.session_state.output_text = ""

    st.session_state.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    write_terminal_output()

def test(file_path, data_dir, query_csv_path, gallery_csv_path, model_opts, checkpoint, batchsize=32, eval_gpu=False):
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

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    return stdout, stderr
