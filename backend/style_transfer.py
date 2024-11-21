import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image, ImageOps
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
import argparse

# 确定运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图片加载和预处理
imsize = 512 if torch.cuda.is_available() else 128

resize_to_imsize = transforms.Compose([
    transforms.Resize(imsize),
    transforms.ToTensor()
])

unloader = transforms.ToPILImage()

def image_loader(image_name):
    """加载图片并调整方向"""
    image = Image.open(image_name)
    image = ImageOps.exif_transpose(image)
    return image

# 内容损失模块
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# 计算Gram矩阵
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

# 风格损失模块
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# 归一化模块
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std

# 加载预训练模型
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

# 创建内容和风格损失模型
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses

# 优化器
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

# 风格迁移主函数
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing...')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Run {run}: Style Loss: {style_score.item():.4f} Content Loss: {content_score.item():.4f}")

            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

# 主程序
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Style Transfer Script")
        parser.add_argument("--content", type=str, required=True, help="Path to content image")
        parser.add_argument("--style", type=str, required=True, help="Path to style image")
        parser.add_argument("--output", type=str, required=True, help="Path to save output image")
        args = parser.parse_args()

        # 加载图片
        content_img = image_loader(args.content)
        style_img = image_loader(args.style)

        # 确保大小一致
        content_img = content_img.resize(style_img.size, Image.LANCZOS)

        content_img = resize_to_imsize(content_img).unsqueeze(0).to(device, torch.float)
        style_img = resize_to_imsize(style_img).unsqueeze(0).to(device, torch.float)

        # 运行风格迁移
        input_img = content_img.clone()
        output = run_style_transfer(cnn, torch.tensor([0.485, 0.456, 0.406]),
                                    torch.tensor([0.229, 0.224, 0.225]),
                                    content_img, style_img, input_img)

        # 保存结果
        output_img = unloader(output.cpu().squeeze(0))
        output_img.save(args.output)
        print(f"Style transfer complete. Output saved to {args.output}")

    except Exception as e:
        print(f"Error occurred: {e}")
        exit(1)
