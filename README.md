# Vehicle Re-identification for Traffic Impact Assessment using Deep Learning

Application with GUI for Vehicle Re-ID tasks build using Streamlit.

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

### Create new virtual environment
```
$ python3 -m venv reid
$ source reid/bin/activate
```

### Install the requirements
```
$ pip3 install -r requirements.txt
```

## Getting Started
Open Streamlit GUI
```
$ streamlit run gui/app.py
```

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

* [regob's Vehicle Re-ID](https://github.com/regob/vehicle_reid)
* [layumi's Re-ID](https://github.com/layumi/Person_reID_baseline_pytorch)

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release