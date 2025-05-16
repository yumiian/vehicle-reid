# Vehicle Re-identification for Traffic Impact Assessment using Deep Learning

A Streamlit-based application designed for Vehicle Re-Identification (Re-ID) tasks, featuring tasks like `Data Preparation`, `Image Comparison`, `Data Augmentation`, `Dataset Split`, `Model Training`, `Model Testing` and `Visualization`.

## Disclaimer

This code only works in **Linux OS**. If you are using Windows, you can use [**WSL (Windows Subsystem for Linux)**](https://learn.microsoft.com/en-us/windows/wsl/install) to install Ubuntu OS on your Windows.

## WSL Installation (for Windows)
**If you already have Linux OS, you can skip this step.**

Install WSL using Windows PowerShell:
```
wsl --install
```

or
```
wsl.exe --install ubuntu
```

If encounter any error, make sure **Windows Subsystem For Linux** is turned on in **Turn Windows Features On and Off**. Then, restart Windows.

Update all packages in Ubuntu:
```
sudo apt update && sudo apt upgrade
```

Make sure Python are installed:
```
sudo apt install python3 python3-pip
```

## Getting Started
First, clone the repo or download the latest source code from [**releases**](https://github.com/yumiian/vehicle-reid/releases).

Create new virtual environment:
```
$ python3 -m venv reid
$ source reid/bin/activate
```

Install [CUDA](https://developer.nvidia.com/cuda-downloads) from Nvidia to utilize the power of GPU to train and test the model. 

Check your installed CUDA version using this command (cmd):
```
nvcc --version
```

Then, install [PyTorch](https://pytorch.org/get-started/locally/) based on your installed CUDA version. Example for installing PyTorch for CUDA version 12.1:
```
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install the required libraries:
```
$ pip3 install -r requirements.txt
```

Finally, open Streamlit GUI:
```
$ streamlit run gui/app.py
```

Now you can view the application in your browser. By default, the app local URL is at http://localhost:8501/.

## Screenshots

### Image Comparison
![compare](https://images2.imgbox.com/96/4a/njnxvpFR_o.png)

### Visualization
![visualization](https://images2.imgbox.com/70/f9/FHVmFYlX_o.png)

## Troubleshoot
This message may showed up after opening the Streamlit GUI using WSL. 

`gio: http://localhost:8501: Operation not supported`

Apparently this issue occurs when WSL is trying to open WSL browser instead of Windows Browser. You can safely ignore this or run this command:

```
$ sudo apt-get install wslu
```

## Acknowledgments

* [regob's Vehicle Re-ID](https://github.com/regob/vehicle_reid)
* [layumi's Re-ID](https://github.com/layumi/Person_reID_baseline_pytorch)
