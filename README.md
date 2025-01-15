# Vehicle Re-identification for Traffic Impact Assessment using Deep Learning

Application with GUI built for Vehicle Re-ID tasks using Streamlit.

## Disclaimer

This code only works in **Linux OS**. If you are using Windows, you can use [**WSL (Windows Subsystem for Linux)**](https://learn.microsoft.com/en-us/windows/wsl/install) to install Ubuntu OS on your Windows.

## Installation
### WSL Installation
Windows PowerShell:
```
wsl --install
```
or
```
wsl.exe --install ubuntu
```
If encounter any error, make sure **Windows Subsystem For Linux** is turned on in **Turn Windows Features On and Off**. Then, restart Windows.

### Update all packages in Ubuntu:
```
sudo apt update && sudo apt upgrade
```

### Make sure Python are installed:
```
sudo apt install python3 python3-pip
```

## Getting Started
First, clone the repo or download the latest source code from [**release**](https://github.com/yumiian/vehicle-reid/releases).

### Create new virtual environment
```
$ python3 -m venv reid
$ source reid/bin/activate
```

### Install the requirements
This requirements.txt is tested on [CUDA](https://developer.nvidia.com/cuda-downloads) version 12.1. If you have different version, please install [PyTorch](https://pytorch.org/get-started/locally/) based on your installed CUDA version.
```
$ pip3 install -r requirements.txt
```

### Check your CUDA version (cmd)
```
nvcc --version
```

### Open Streamlit GUI
```
$ streamlit run gui/app.py
```

## Acknowledgments

* [regob's Vehicle Re-ID](https://github.com/regob/vehicle_reid)
* [layumi's Re-ID](https://github.com/layumi/Person_reID_baseline_pytorch)