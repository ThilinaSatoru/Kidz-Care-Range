# Define paths
import os

is_win = False
# Check if the system is Linux or Windows
if os.name == 'nt':
    is_win = True

VIDEO_PATH = r"samples/v1.mp4"
MODEL_PATH = r"YOLO/94.1 63.3/model.pt"

OUTPUT_DIRECTORY = 'OUTS'
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
