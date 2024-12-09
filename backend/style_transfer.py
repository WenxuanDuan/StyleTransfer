import torch
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import argparse
import utils  # 引入工具模块
from network import ImageTransformNet  # 引入快速风格迁移网络

# 确定运行设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图片加载和预处理
imsize = 512 if torch.cuda.is_available() else 128

resize_to_imsize = transforms.Compose([
    transforms.Resize(imsize),
    transforms.CenterCrop(imsize),
    transforms.ToTensor(),
    utils.normalize_tensor_transform()
])

unloader = transforms.ToPILImage()

def image_loader(image_name):
    """加载图片并调整方向"""
    image = Image.open(image_name)
    image = ImageOps.exif_transpose(image)
    return image

def run_fast_style_transfer(content_img_path, model_path, output_img_path):
    """运行快速风格迁移"""
    # 加载内容图片
    content_img = image_loader(content_img_path)
    content_img = resize_to_imsize(content_img).unsqueeze(0).to(device, torch.float)

    # 加载预训练模型
    style_model = ImageTransformNet()
    style_model.load_state_dict(torch.load(model_path))
    style_model = style_model.to(device)
    style_model.eval()

    # 应用风格迁移
    with torch.no_grad():
        output = style_model(content_img).cpu().squeeze(0)
        output_img = unloader(output)

    # 保存结果
    output_img.save(output_img_path)
    print(f"Fast style transfer complete. Output saved to {output_img_path}")

# 主程序
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Fast Style Transfer Script")
        parser.add_argument("--content", type=str, required=True, help="Path to content image")
        parser.add_argument("--model", type=str, required=True, help="Path to pre-trained model")
        parser.add_argument("--output", type=str, required=True, help="Path to save output image")
        args = parser.parse_args()

        # 执行快速风格迁移
        run_fast_style_transfer(args.content, args.model, args.output)

    except Exception as e:
        print(f"Error occurred: {e}")
        exit(1)
