"""
Neural Transfer Using PyTorch
=============================
**Author**: `Alexis Jacq <https://alexis-jacq.github.io>`_
**Edited by**: `Winston Herring <https://github.com/winston6>`_

Introduction
------------
本教程解释如何实现由 Leon A. Gatys、Alexander S. Ecker 和 Matthias Bethge 开发的神经风格算法 <https://arxiv.org/abs/1508.06576>。
神经风格，或神经风格迁移，允许你采用一张图像并以新的艺术风格复制它。
该算法采用三张图像，一张输入图像、一张内容图像和一张风格图像，并更改输入以使其类似于内容图像的内容和风格图像的艺术风格。
"""

######################################################################
'''
原理很简单：我们定义两个距离，一个用于内容(D_c)，另一个用于风格(D_s)。 
D_c衡量两张图像之间内容的差异，而D_s衡量两张图像之间风格的差异。
然后，我们取第三张图像，即输入图像，并将其转换以最小化其与内容图像的内容距离及其与风格图像的风格距离。

现在我们可以导入必要的包并开始神经风格迁移：
torch、torch.nn、numpy（PyTorch 中神经网络必不可少的包）
torch.optim（高效的梯度下降）
PIL、PIL.Image、matplotlib.pyplot（加载和显示图像）
torchvision.transforms（将 PIL 图像转换为张量）
torchvision.models（训练或加载预训练模型）
copy（深度复制模型；系统包）
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image, ImageOps
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

import copy

######################################################################
'''
接下来，我们需要选择在哪个设备上运行网络并导入内容和风格图像。
在大型图像上运行神经风格迁移算法需要更长的时间，而在 GPU 上运行速度会快得多。
我们可以使用 torch.cuda.is_available() 检测是否有 GPU 可用。
接下来，我们设置 torch.device 以便在整个教程中使用。
此外，.to(device) 方法用于将张量或模块移动到所需的设备。
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

######################################################################
# 加载图像
'''
现在我们将导入风格和内容图像。
原始的PIL图像的像素值介于 0 和 255 之间，但当转换为 torch 张量时，它们的值会被转换为介于 0 和 1 之间。（通过除以255实现）
图像还需要调整大小以具有相同的尺寸。
需要注意的一个重要细节是，torch 库中的神经网络是用 0 到 1 范围内的张量值进行训练的。
如果您尝试向网络馈送 0 到 255 的张量图像，则激活的特征图将无法感知预期的内容和风格。
但是，来自 Caffe 库的预训练网络是用 0 到 255 的张量图像进行训练的。

picasso.jpg <https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg>
dancing.jpg <https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg>
'''

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU

# loader： 用于将图像缩放到指定尺寸并转换为张量
resize_to_imsize = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

# loader：只转换为张量
to_tensor = transforms.ToTensor()

# 用于加载和预处理图像
def image_loader(image_name):
    # 此时，image是一个PIL图像对象
    image = Image.open(image_name)
    image = ImageOps.exif_transpose(image)
    return image

style_img = image_loader("./images/style/StarryNight.jpg")
content_img = image_loader("./images/content/SpaceNeedle.jpg")

# 将content图像缩放成style图像的大小
content_img = content_img.resize(style_img.size, Image.LANCZOS)

# 将缩放后的style_img和content_img都缩放到目标imsize
style_img = resize_to_imsize(style_img).unsqueeze(0).to(device, torch.float)
content_img = resize_to_imsize(content_img).unsqueeze(0).to(device, torch.float)

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"

######################################################################
'''
现在，让我们创建一个函数，通过将其副本重新转换为 PIL 格式并在使用 plt.imshow 显示副本的方式来显示图像。
我们将尝试显示内容和风格图像以确保它们已正确导入。
'''
unloader = transforms.ToPILImage()  # reconvert into PIL image
plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.figure()
imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')

######################################################################
# 损失函数
# --------------
# 内容损失
'''
内容损失是一个函数，它表示单个层的加权内容距离。
该函数获取网络处理输入X的第L层的特征图F_{XL}，并返回图像X和内容图像C之间的加权内容距离（weighted content distance）: w_{CL}.D_C^L(X,C)。
该函数必须知道内容图像的特征图，F_{CL}，才能计算内容距离。
我们将此函数实现为一个torch模块，其构造函数将F_{CL}作为输入。
距离，\|F_{XL} - F_{CL}\|^2，是两组特征图之间的均方误差，可以使用nn.MSELoss计算。

我们将此内容损失模块直接添加到用于计算内容距离的卷积层之后。
这样，每次网络馈送输入图像时，都会在所需的层计算内容损失；并且，由于自动梯度，所有梯度都将被计算。
现在，为了使内容损失透明，我们必须定义一个forward方法，该方法计算内容损失，然后返回层的输入。
计算出的损失被保存为模块的参数。

**重要细节**：
尽管此模块名为 ContentLoss，但它不是真正的 PyTorch 损失函数。
如果要将内容损失定义为 PyTorch 损失函数，则必须创建一个 PyTorch 自动梯度函数，以在 backward 方法中手动重新计算/实现梯度。
'''
class ContentLoss(nn.Module):

    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # target是内容图像经过CNN处理后在某一层的特征图
        # detach()用于将张量从计算图中分离出来，使其成为一个固定的值，不再参与自动求导
        # 这样做的目的是告诉网络，target是一个常量，不需要计算梯度
        self.target = target.detach()

    def forward(self, input):
        # input是当前生成图像在该层的特征图
        # 计算input和target之间的均方误差MSE
        self.loss = F.mse_loss(input, self.target)
        # 返回input，这使得ContentLoss层在网络中“透明”，不改变数据流，只是计算并记录内容损失。
        return input

######################################################################
# 风格损失
'''
风格损失用于衡量生成图像与目标风格图像在风格上的相似性。
风格损失通过计算生成图像和风格图像在某一特征层的格拉姆矩阵，Gram Matrix，G_{XL}，之间的差异来实现。
格拉姆矩阵是将给定矩阵与其转置矩阵相乘的结果，其中的每个元素表示一对特征图之间的相关性。
因此，格拉姆矩阵包含了特征图的协作关系，这些关系能够反映图像的风格特征，而不依赖于图像的具体位置或内容。

在此应用中，给定矩阵是第L层的特征图F_{XL}的重塑版本。
F_{XL}将被重塑成\hat{F}_{XL}，这是一个K*N矩阵，其中K是第L层的特征图的数量，N是任何矢量化特征图F_{XL}^k的长度。
例如，\hat{F}_{XL}的第一行对应于第一个矢量化特征图F_{XL}^1。

最后，必须通过将每个元素除以矩阵中的元素总数来对格拉姆矩阵进行归一化（normalization）。
这种归一化是为了抵消\hat{F}_{XL}矩阵具有较大N维度会在格拉姆矩阵中产生较大值的事实。
这些较大的值会导致第一层（池化层之前）在梯度下降期间产生更大的影响。
风格特征往往位于网络的更深层，因此此归一化步骤至关重要。

风格损失模块的实现类似于内容损失模块。它将作为网络中计算该层风格损失的透明层。
风格距离也使用G_{XL}和 G_{SL}之间的均方误差来计算。
'''

def gram_matrix(input):
    # input是该层特征图的张量，尺寸为(a, b, c, d)
    # a是批量大小（通常为1，因为我们处理单张图像
    # b是特征图的数量，即该层的通道数
    # c和d是特征图的高度和宽度，N = c * d，表示特征图的像素数量
    a, b, c, d = input.size()

    # 将input重塑为一个大小为(a * b, c * d)的2D张量。每行代表一个特征图（通道），每列代表特征图的所有像素展开为一维。
    features = input.view(a * b, c * d)  # resize F_XL into \hat F_XL

    # G，格拉姆矩阵，表示每对特征图之间的相关性
    G = torch.mm(features, features.t())  # compute the gram product

    # 对G进行归一化，防止由于特征图较大的尺寸而导致值过大。这种归一化可以使不同层次的特征图对风格损失的贡献保持一致。
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        # target_feature，代表目标风格图像在特定层的特征图
        # 计算target_feature的格拉姆矩阵，并将其与计算图分离，使其成为常量，以避免自动求导过程中对target计算梯度。
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        # input表示当前生成图像在特定层的特征图
        # 计算input的格拉姆矩阵
        G = gram_matrix(input)
        # 计算生成图像的G和目标图像的G之前的均方误差
        self.loss = F.mse_loss(G, self.target)
        # 返回input，保持该层的透明性，不改变数据流
        return input

######################################################################
# 导入模型
'''
现在我们需要导入一个预训练的神经网络。我们将使用类似于论文中使用的 19 层 VGG 网络。
PyTorch 中 VGG 的实现是一个模块，它分为两个子 Sequential 模块：features（包含卷积和池化层）和 classifier（包含全连接层）。
我们将使用 features 模块，因为我们需要各个卷积层的输出来测量内容和风格损失。
某些层在训练期间的行为与评估期间不同，因此我们必须使用 .eval() 将网络设置为评估模式。
'''
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

######################################################################
'''
VGG19 网络是在归一化后的图像上进行训练的，因此在将输入图像送入网络前，需要对图像进行相同的归一化操作。

mean=[0.485, 0.456, 0.406] 和 std=[0.229, 0.224, 0.225] 
分别是 VGG19 模型在 ImageNet 数据集上训练时用到的均值和标准差，分别用于 R、G、B 三个通道的归一化。
mean：图像的每个通道分别减去相应的均值。
std：然后每个通道除以相应的标准差。

这将像素值归一化到标准范围，使得图像特征更符合模型的训练分布。

'''
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

# create a module to normalize input image so we can easily put it in a ``nn.Sequential``
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std


######################################################################
'''
一个 Sequential 模块包含一个按顺序排列的子模块列表。
例如，vgg19.features 包含一个按深度顺序排列的序列（Conv2d、ReLU、MaxPool2d、Conv2d、ReLU……）。
我们需要在它们检测到的卷积层之后立即添加内容损失和风格损失层。
为此，我们必须创建一个新的 Sequential 模块，该模块正确插入了内容损失和风格损失模块。
'''
# desired depth layers to compute style/content losses :
# 在conv_4层计算内容损失
content_layers_default = ['conv_4']
# 用conv_1,...,conv_5计算风格损失
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# 构建一个新的模型，它包含VGG19网络的卷积层、归一化层，以及在指定层插入的内容损失和风格损失模块
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module，对图像进行归一化
    normalization = Normalization(normalization_mean, normalization_std)

    # just in order to have an iterable access to or list of content/style losses
    content_losses = []
    style_losses = []

    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    # model将包含VGG19网络的各层以及内容和风格损失模块
    # 将normalization模块作为第一个模块添加进去
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    # 使用cnn.children()遍历VGG19的每一层，将其逐一添加到model中
    for layer in cnn.children():
        # 为每个层生成名称
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        # 将当前层添加到model中，并赋予指定的名称
        model.add_module(name, layer)

        # 如果当前层在content_layers中，则在此层后添加一个ContentLoss模块
        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)
        # 如果当前层在style_layers中，则在此层后添加一个ContentLoss模块
        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # 从最后一层向前遍历model，找到最后一个ContentLoss或StyleLoss层，然后裁剪掉该层之后的所有层
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses


######################################################################
#接下来，我们选择输入图像。可以使用内容图像的副本或白噪声。

input_img = content_img.clone()
# if you want to use white noise by using the following code:
# input_img = torch.randn(content_img.data.size())

# add the original input image to the figure:
plt.figure()
imshow(input_img, title='Input Image')


######################################################################
# 梯度下降
'''
正如算法作者Leon Gatys建议的那样，我们将使用L-BFGS算法来运行我们的梯度下降。
<https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq>

与训练网络不同，我们希望训练输入图像以最小化内容/风格损失。
我们将创建一个PyTorch L-BFGS优化器optim.LBFGS并将我们的图像作为要优化的张量传递给它。
'''
def get_input_optimizer(input_img):
    # input_img是生成的图像
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer

######################################################################
'''
最后，我们必须定义一个执行神经风格迁移的函数。
对于网络的每次迭代，都会向其馈送更新的输入并计算新的损失。
我们将运行每个损失模块的backward方法以动态计算它们的梯度。
优化器需要一个“闭包”函数，该函数会重新评估模块并返回损失。

我们还有一个最终约束需要解决。
网络可能会尝试优化输入，其值超过图像的0到1张量范围。
我们可以通过每次运行网络时将输入值校正为0到1之间来解决问题。
'''

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):

    # num_steps是优化过程中的迭代次数，默认值为300
    # style_weight和content_weight是风格和内容损失的权重，通常风格权重较大，以确保生成图像更接近风格图像的风格

    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img,
                                                                     content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1) # 将 input_img 的像素值限制在 0 到 1 之间

            optimizer.zero_grad() # 清除之前的梯度，以免梯度累积。
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
            loss.backward() # 反向传播，计算 input_img 的梯度。

            run[0] += 1
            # 每 50 次迭代输出当前的迭代次数、风格损失和内容损失，帮助我们跟踪优化过程。
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


######################################################################
# 最后，我们可以运行算法。
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()

