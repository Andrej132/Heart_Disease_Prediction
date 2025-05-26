import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
MODEL_DIR = os.path.join(ARTIFACTS_DIR, 'models')
IMAGES_DIR = os.path.join(ARTIFACTS_DIR, 'images')
LOG_DIR = os.path.join(BASE_DIR, 'logs')