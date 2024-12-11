import numpy as np
import torch
import os
import argparse
import time

from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import utils
from network import ImageTransformNet
from vgg import Vgg16

# Global Variables
IMAGE_SIZE = 512
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
EPOCHS = 2
STYLE_WEIGHT = 1e11
CONTENT_WEIGHT = 1e5
TV_WEIGHT = 1e-6

def get_device(gpu=None):
    """
    Determine the device to use (MPS, CUDA, or CPU).

    Args:
        gpu (int): GPU ID to use for CUDA. If None, checks for MPS or falls back to CPU.

    Returns:
        torch.device: The selected device.
    """
    if gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        print(f"Using CUDA on GPU {gpu}.")
        return torch.device(f"cuda:{gpu}")
    elif torch.backends.mps.is_available():
        print("Using Metal Performance Shaders (MPS) for computation.")
        return torch.device("mps")
    else:
        print("Using CPU for computation.")
        return torch.device("cpu")

def train(args):
    # Get the device
    device = get_device(args.gpu)

    # visualization of training controlled by flag
    visualize = args.visualize is not None
    if visualize:
        img_transform_512 = transforms.Compose([
            transforms.Resize(512),  # scale shortest side to image_size
            transforms.CenterCrop(512),  # crop center image_size out
            transforms.ToTensor(),  # turn image from [0-255] to [0-1]
            utils.normalize_tensor_transform()  # normalize with ImageNet values
        ])

        testImage_amber = utils.load_image("content/amber.jpg")
        testImage_amber = img_transform_512(testImage_amber).unsqueeze(0).to(device)

    # define network
    image_transformer = ImageTransformNet().to(device)
    optimizer = Adam(image_transformer.parameters(), LEARNING_RATE)

    loss_mse = torch.nn.MSELoss()

    # load vgg network
    vgg = Vgg16().to(device)

    # get training dataset
    dataset_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),  # scale shortest side to image_size
        transforms.CenterCrop(IMAGE_SIZE),  # crop center image_size out
        transforms.ToTensor(),  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()  # normalize with ImageNet values
    ])
    train_dataset = datasets.ImageFolder(args.dataset, dataset_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

    # style image
    style_transform = transforms.Compose([
        transforms.ToTensor(),  # turn image from [0-255] to [0-1]
        utils.normalize_tensor_transform()  # normalize with ImageNet values
    ])
    style = utils.load_image(args.style_image)
    style = style_transform(style).unsqueeze(0).to(device)
    style_name = os.path.split(args.style_image)[-1].split('.')[0]

    # calculate gram matrices for style feature layer maps we care about
    style_features = vgg(style)
    style_gram = [utils.gram(fmap) for fmap in style_features]

    for e in range(EPOCHS):
        img_count = 0
        aggregate_style_loss = 0.0
        aggregate_content_loss = 0.0
        aggregate_tv_loss = 0.0

        # train network
        image_transformer.train()
        for batch_num, (x, _) in enumerate(train_loader):
            img_batch_read = len(x)
            img_count += img_batch_read

            # zero out gradients
            optimizer.zero_grad()

            # input batch to transformer network
            x = x.to(device)
            y_hat = image_transformer(x)

            # get vgg features
            y_c_features = vgg(x)
            y_hat_features = vgg(y_hat)

            # calculate style loss
            y_hat_gram = [utils.gram(fmap) for fmap in y_hat_features]
            style_loss = 0.0
            for j in range(4):
                # Match the batch size of the style_gram with y_hat_gram
                style_gram_batch = style_gram[j].repeat(img_batch_read, 1, 1)
                style_loss += loss_mse(y_hat_gram[j], style_gram_batch)
            style_loss = STYLE_WEIGHT * style_loss
            aggregate_style_loss += style_loss.item()

            # calculate content loss (h_relu_2_2)
            recon = y_c_features[1]
            recon_hat = y_hat_features[1]
            content_loss = CONTENT_WEIGHT * loss_mse(recon_hat, recon)
            aggregate_content_loss += content_loss.item()

            # calculate total variation regularization
            diff_i = torch.sum(torch.abs(y_hat[:, :, :, 1:] - y_hat[:, :, :, :-1]))
            diff_j = torch.sum(torch.abs(y_hat[:, :, 1:, :] - y_hat[:, :, :-1, :]))
            tv_loss = TV_WEIGHT * (diff_i + diff_j)
            aggregate_tv_loss += tv_loss.item()

            # total loss
            total_loss = style_loss + content_loss + tv_loss

            # backprop
            total_loss.backward()
            optimizer.step()

            # print out status message
            if ((batch_num + 1) % 100 == 0):
                print(
                    f"{time.ctime()}  Epoch {e + 1}:  [{img_count}/{len(train_dataset)}]  "
                    f"style_loss: {style_loss.item():.6f}  content_loss: {content_loss.item():.6f}  tv_loss: {tv_loss.item():.6f}"
                )

    # save model
    image_transformer.eval()

    if not os.path.exists("models"):
        os.makedirs("models")
    filename = f"models/{style_name}_{time.ctime().replace(' ', '_')}.model"
    torch.save(image_transformer.state_dict(), filename)


def style_transfer(args):
    # Get the device
    device = get_device(args.gpu)

    # Load content image
    content = utils.load_image(args.source)
    content = transforms.ToTensor()(content).unsqueeze(0).to(device)

    # Load style model
    style_model = ImageTransformNet().to(device)
    style_model.load_state_dict(torch.load(args.model_path, map_location=device))
    style_model.eval()

    # Apply style transfer
    with torch.no_grad():
        output = style_model(content).cpu().squeeze(0)
        utils.save_image(args.output, output)

    print(f"Style transfer complete. Output saved to {args.output}")

def blend_styles(args):
    device = get_device(args.gpu)

    # Load content image
    content = utils.load_image(args.source)
    content = transforms.ToTensor()(content).unsqueeze(0).to(device)

    # Load style models
    style_model1 = ImageTransformNet().to(device)
    style_model1.load_state_dict(torch.load(args.model_path1, map_location=device))
    style_model1.eval()

    style_model2 = ImageTransformNet().to(device)
    style_model2.load_state_dict(torch.load(args.model_path2, map_location=device))
    style_model2.eval()

    # Apply styles with blending
    with torch.no_grad():
        output1 = style_model1(content).cpu()
        output2 = style_model2(content).cpu()
        blended_output = args.weight * output1 + (1 - args.weight) * output2

    utils.save_image(args.output, blended_output.squeeze(0))
    print(f"Blended style transfer complete. Output saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Style transfer in PyTorch")
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train a model to do style transfer")
    train_parser.add_argument("--style-image", type=str, required=True, help="Path to a style image to train with")
    train_parser.add_argument("--dataset", type=str, required=True, help="Path to a dataset")
    train_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")
    train_parser.add_argument("--visualize", type=int, default=None, help="Set to 1 if you want to visualize training")

    # Transfer subcommand
    transfer_parser = subparsers.add_parser("transfer", help="Do style transfer with a trained model")
    transfer_parser.add_argument("--model-path", type=str, required=True, help="Path to a pretrained model")
    transfer_parser.add_argument("--source", type=str, required=True, help="Path to source image")
    transfer_parser.add_argument("--output", type=str, required=True, help="Path to save the stylized output")
    transfer_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")

    # Blend subcommand
    blend_parser = subparsers.add_parser("blend", help="Blend two styles")
    blend_parser.add_argument("--model-path1", type=str, required=True, help="Path to the first style model")
    blend_parser.add_argument("--model-path2", type=str, required=True, help="Path to the second style model")
    blend_parser.add_argument("--weight", type=float, required=True, help="Weight for the first style (0-1)")
    blend_parser.add_argument("--source", type=str, required=True, help="Path to source image")
    blend_parser.add_argument("--output", type=str, required=True, help="Path to save the blended output")
    blend_parser.add_argument("--gpu", type=int, default=None, help="ID of GPU to be used")

    args = parser.parse_args()

    if args.subcommand == "train":
        print("Training!")
        train(args)
    elif args.subcommand == "transfer":
        print("Style transferring!")
        style_transfer(args)
    elif args.subcommand == "blend":
        print("Blending styles!")
        blend_styles(args)
    else:
        print("Invalid command")

if __name__ == "__main__":
    main()







