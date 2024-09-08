# Define paths
import os
import platform

# Check if the system is Linux or Windows
is_win = platform.system() == "Windows"
is_linux = platform.system() == "Linux"

VIDEO_PATH = r"samples/v2.mp4"
MODEL_PATH = r"YOLO/94.1 63.3/model.pt"

OUTPUT_DIRECTORY = 'OUTS'
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
