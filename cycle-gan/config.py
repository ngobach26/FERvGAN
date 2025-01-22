import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

LOG_PATH = "logs"
MLFLOW_EXP = 'identity_loss'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAME = "fer"
TRAIN_DIR = f"datasets/{DATASET_NAME}/train"
VAL_DIR = f"datasets/{DATASET_NAME}/test"
IMG_SAVE_INTERVAL = 5
BATCH_SIZE = 1
GEN_LEARNING_RATE = 1e-5
DIS_LEARNING_RATE = 5e-6
LAMBDA_IDENTITY = 0.0
CYCLE_LOSS_COEFFICIENT = 5
LAMBDA_GEN_IDENTITY = 1.0
N_BLOCKS = 6
IN_CHANNELS = 1
NUM_WORKERS = 8
NUM_EPOCHS = 200
TRAIN = True
LOAD_MODEL = False
SAVE_MODEL = True
TEST_EPOCHS = 1
REFERENCE_NAME = 'neutral'
TARGET_NAME = 'disgust'
CHECKPOINT_GEN_R = "gen_tr.pth.tar"
CHECKPOINT_GEN_T = "gen_rt.pth.tar"
CHECKPOINT_DISC_R = "disc_r.pth.tar"
CHECKPOINT_DISC_T = "disc_t.pth.tar"
SAVE_IMAGE_PATH = 'save_images'
IMG_w, IMG_H = 128, 128
MEAN, STD = [0.5], [0.5]
transforms = A.Compose(
    [
        A.Resize(width=IMG_w, height=IMG_H),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
