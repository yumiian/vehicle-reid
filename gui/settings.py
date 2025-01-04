from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

MODEL_DIR = ROOT / 'models'
YOLO_MODEL = MODEL_DIR / 'yolo11n.pt'

VIDEO_DIR = ROOT / 'videos'
VIDEO_FILE = VIDEO_DIR / 'video.mp4'

OUTPUT_DIR = ROOT / 'results'
NEW_RESULT_DIR = ROOT / 'new_results'
DATASETS_DIR = ROOT / "datasets"