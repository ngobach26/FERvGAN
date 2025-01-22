import os
from os.path import join
import torch
import torchvision.transforms as transforms
import numpy as np
from dataset import ReferenceTargetDataset
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
# import mlflow
from torchvision.utils import save_image
import torchvision.transforms.functional as TF
from discriminator_model import Discriminator
from generator_model import Generator
from deep_utils import show_destroy_cv2, get_logger
from deep_utils import ModelCheckPointTorch, log_print, CSVLogger, TensorboardTorch, mkdir_incremental



def test_model(disc_R, disc_T, gen_RT, gen_TR, test_loader, device, save_dir):
    """
    Function to test the model on a test dataset.
    Args:
        disc_R, disc_T: Discriminator models (optional, depending on use case)
        gen_RT, gen_TR: Generator models
        test_loader: DataLoader for the testing dataset
        device: Device to run the inference on
        save_dir: Directory to save generated images
    """
    gen_RT.eval()
    gen_TR.eval()
    disc_R.eval()
    disc_T.eval()

    os.makedirs(save_dir, exist_ok=True)

    comparison_dir = os.path.join(save_dir, "comparison")
    generated_fake_dir = os.path.join(save_dir, "generated_fake")
    os.makedirs(comparison_dir, exist_ok=True)
    os.makedirs(generated_fake_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (target, reference) in enumerate(test_loader):
            target = target.to(device)
            reference = reference.to(device)

            # Generate fake images
            fake_reference = gen_TR(target)
            fake_target = gen_RT(reference)

            # Normalize images for visualization
            target = target * 0.5 + 0.5
            reference = reference * 0.5 + 0.5
            fake_reference = fake_reference * 0.5 + 0.5
            fake_target = fake_target * 0.5 + 0.5

            # Resize fake target image to 48x48
            fake_target_resized = TF.resize(fake_target, (48, 48))

            # Concatenate images horizontally
            concat_image = torch.cat((target, reference, fake_reference, fake_target), dim=3)

            # Save concatenated image in the comparison folder
            save_image(concat_image, f"{comparison_dir}/concat_{idx}.png")

            # Save resized fake target image in the generated_fake folder
            save_image(fake_target_resized, f"{generated_fake_dir}/fake_target_{idx}.png")

    print(f"Concatenated images saved in {comparison_dir}")
    print(f"Fake target images saved in {generated_fake_dir}")


def main(save_dir=''):
    # Load models
    disc_R = Discriminator(in_channels=config.IN_CHANNELS).to(config.DEVICE)
    disc_T = Discriminator(in_channels=config.IN_CHANNELS).to(config.DEVICE)
    gen_RT = Generator(img_channels=config.IN_CHANNELS, num_residuals=config.N_BLOCKS).to(config.DEVICE)
    gen_TR = Generator(img_channels=config.IN_CHANNELS, num_residuals=config.N_BLOCKS).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_R.parameters()) + list(disc_T.parameters()),
        lr=config.DIS_LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_RT.parameters()) + list(gen_TR.parameters()),
        lr=config.GEN_LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    # Load the saved model weights
    load_checkpoint(
        join(save_dir, config.CHECKPOINT_GEN_R), gen_TR, opt_gen, config.GEN_LEARNING_RATE,
    )
    load_checkpoint(
        join(save_dir, config.CHECKPOINT_GEN_T), gen_RT, opt_gen, config.GEN_LEARNING_RATE,
    )
    load_checkpoint(
        join(save_dir, config.CHECKPOINT_DISC_R), disc_R, opt_disc, config.DIS_LEARNING_RATE,
    )
    load_checkpoint(
        join(save_dir, config.CHECKPOINT_DISC_T), disc_T, opt_disc, config.DIS_LEARNING_RATE,
    )


    # Load test dataset
    test_dataset = ReferenceTargetDataset(
        root_reference=config.TRAIN_DIR + f"/{config.REFERENCE_NAME}",
        root_target=config.TRAIN_DIR + f"/{config.TARGET_NAME}",
        transform=config.transforms
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True
    )

    # Test the model
    output_dir = "outputs"
    test_model(disc_R, disc_T, gen_RT, gen_TR, test_loader, config.DEVICE, output_dir)


if __name__ == "__main__":
    main()
