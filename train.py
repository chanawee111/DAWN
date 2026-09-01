import os
import json
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision

from utilities import load_train_setting
from models.model import *
from utilities.load_train_setting import *
from tqdm import tqdm

# ------------------------------------------------
# Setup
# ------------------------------------------------
# Initialize training folders and create result directory
result_folder = initialize_training_folders()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize TensorBoard
writer = SummaryWriter(log_dir=os.path.join(result_folder, "tensorboard_logs"))

# Initialize model
model = Model(H, W, message_length, noise_layers, device, batch_size, lr, decoder_only_training=decoder_only_training, seed=seed)

# Load dataset with augmentation
train_dataset = Dataset_loader_augmentation(os.path.join(dataset_path, "coco_color_filter_ratio_20000"), H, W)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

val_dataset = Dataset_loader(os.path.join(dataset_path, "val2017_2500"), H, W)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

def log_module_params(module, module_name, epoch):
    for name, param in module.named_parameters():
        full_name = f"{module_name}/{name}"
        writer.add_histogram(f"Weights/{full_name}", param.data.cpu(), epoch)
        if param.grad is not None:
            writer.add_histogram(f"Gradients/{full_name}", param.grad.cpu(), epoch)

# save trainable parameters and total parameters
model.save_trainable_parameters(result_folder)  

# Load from checkpoint if needed
latest_epoch = 0
if train_continue:
    model_dir = os.path.join("results", train_continue_path, "models")
    encoder_files = [f for f in os.listdir(model_dir) if f.startswith("Encoder_") and f.endswith(".pth")]
    
    if encoder_files:
        # Extract epoch numbers
        if train_continue_from_lastest_epoch:
            epochs = [int(f.split("_")[1].split(".")[0]) for f in encoder_files]
            latest_epoch = max(epochs)
        
        # Construct paths
        print("train_continue_epoch:", train_continue_epoch)
        Encoder_model_path = os.path.join(model_dir, f"Encoder_{latest_epoch if train_continue_from_lastest_epoch else train_continue_epoch}.pth")
        Discriminator_path = os.path.join(model_dir, f"Discriminator_{latest_epoch if train_continue_from_lastest_epoch else train_continue_epoch}.pth")
        latest_epoch = latest_epoch if train_continue_from_lastest_epoch else train_continue_epoch
        print(f"Encoder_model_path: {Encoder_model_path}")
        print(f"Discriminator_path: {Discriminator_path}")
        model.load_model(Encoder_model_path, Discriminator_path)
        print(f"✔ Resuming from checkpoint at epoch {latest_epoch}")
    else:
        print("⚠ train_continue is True, but no checkpoints found. Starting from scratch.")

print("=" * 20)
print("Training Start")
print("=" * 20)

# ------------------------------------------------
# Training Loop
# ------------------------------------------------
for epoch in range(latest_epoch + 1, epoch_number + 1):
    # epoch += train_continue_epoch if train_continue else 0

    training_result = {
        "error_rate": 0.0,
        "psnr_loss": 0.0,
        "ssim_loss": 0.0,
        "generator_loss": 0.0,
        "generator_loss_on_discriminator": 0.0,
        "generator_loss_on_encoder": 0.0,
        "generator_loss_on_decoder": 0.0,
        "discriminator_cover_loss": 0.0,
        "discriminator_encoded_loss": 0.0,
        "decode_accuracy" : 0.0,
        "Lpips_loss_in_train" : 0.0
    }

    start_time = time.time()
    num = 0
    # ------------------------------------------------
    # Training
    # ------------------------------------------------
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch} [Train]")
    for images in pbar:
        image = images.to(device)

        
        # print("image range:", image.min().item(), image.max().item())
        message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

        result, messages, decoded_message = model.train(image, message, epoch) if not decoder_only_training else model.train_decoder(image, message, epoch)

        for key in result:
            training_result[key] += float(result[key])

        num += 1
        pbar.set_postfix({key: f"{training_result[key]/num:.4f}" for key in ["error_rate","decode_accuracy", "psnr_loss", "ssim_loss","Lpips_loss_in_train"]})

    # model.scheduler_ed.step()
    # model.scheduler_disc.step()

    if hasattr(model, 'encoder_decoder'):
        log_module_params(model.encoder_decoder, "encoder_decoder", epoch)
    if hasattr(model, 'discriminator'):
        log_module_params(model.discriminator, "Discriminator", epoch)

    for key in training_result:
        writer.add_scalar(f"Train/{key}", training_result[key] / num, epoch)

    for i in range(image.shape[0]):  
        original_msg = message[i].cpu().numpy().astype(int)
        decoded_msg = decoded_message[i].detach().cpu().numpy().round().astype(int)

        writer.add_text(f"Message/original_{i}", str(original_msg), epoch)
        writer.add_text(f"Message/decoded_{i}", str(decoded_msg), epoch)

    # Save training log
    content = f"Epoch {epoch} : {int(time.time() - start_time)}s\n"
    for key in training_result:
        content += f"{key}={training_result[key] / num},"
    content += "\n"

    with open(os.path.join(result_folder, "train_log.txt"), "a") as file:
        file.write(content)
    print(content)

    weights_content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
                                                            time.localtime()) + f" | Epoch {epoch} -----------------------\n"

    # ------------------------------------------------
    # Validation
    # ------------------------------------------------
    val_result = {
        "error_rate": 0.0,
        "psnr_loss": 0.0,
        "ssim_loss": 0.0,
        "generator_loss": 0.0,
        "generator_loss_on_discriminator": 0.0,
        "generator_loss_on_encoder": 0.0,
        "generator_loss_on_decoder": 0.0,
        "discriminator_cover_loss": 0.0,
        "discriminator_encoded_loss": 0.0,
        "decode_accuracy" : 0.0,
        "Lpips_loss_in_train" : 0.0      
    }

    start_time = time.time()
    saved_iterations = np.random.choice(np.arange(len(val_dataloader)), size=save_images_number, replace=False)
    saved_all = None
    num = 0

    pbar = tqdm(val_dataloader, desc=f"Epoch {epoch} [Val]")
    for i, images in enumerate(pbar):
        image = images.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

        result, (images, encoded_images, noised_image, messages, decoded_message) = model.validation(image, message, epoch)

        for key in result:
            val_result[key] += float(result[key])
        num += 1

        if i in saved_iterations:
            if saved_all is None:
                saved_all = get_random_images(images, encoded_images, noised_image)
            else:
                saved_all = concatenate_images(saved_all, image, encoded_images, noised_image)

        pbar.set_postfix({key: f"{val_result[key]/num:.4f}" for key in ["error_rate","decode_accuracy", "psnr_loss", "ssim_loss","Lpips_loss_in_train"]})

    for key in val_result:
        writer.add_scalar(f"Val/{key}", val_result[key] / num, epoch)

    # Save image grid to TensorBoard
    if saved_all is not None:
        if epoch % 10 == 0:
            titles = ["Original", "Encoded", "Noised"]
            for idx, img_set in enumerate(saved_all):
                grid = torchvision.utils.make_grid(img_set, normalize=True, scale_each=True)
                writer.add_image(f"Validation/{titles[idx]}", grid, epoch)

    # Save validation log
    content = f"Epoch {epoch} : {int(time.time() - start_time)}s\n"
    for key in val_result:
        content += f"{key}={val_result[key] / num},"
    content += "\n"

    with open(os.path.join(result_folder, "val_log.txt"), "a") as file:
        file.write(content)
    print(content)

    # Save model
    path_model = os.path.join(result_folder, "models")
    os.makedirs(path_model, exist_ok=True)

    path_encoder_decoder = os.path.join(path_model, f"Encoder_{epoch}.pth")
    path_discriminator = os.path.join(path_model, f"Discriminator_{epoch}.pth")
    model.save_model(path_encoder_decoder, path_discriminator)

# Close TensorBoard writer at the end
writer.close()
