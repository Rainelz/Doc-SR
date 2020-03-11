import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import load_pretrained_weights, adv_prop_transform
from PIL import Image


class Discriminator_EfficientNet(nn.Module):  # nn.module
    def __init__(self, model_name, in_ch=3, pretrained=None, drop_fc=False, use_input_norm=False):
        super(Discriminator_EfficientNet, self).__init__()
        self.use_input_norm = use_input_norm
        self.image_size = EfficientNet.get_image_size(model_name)  # 224
        grayscale = 'gray' in model_name
        if grayscale:
            means = [0.5]
            stds = [0.5]
        else:
            means = [0.485, 0.456, 0.406]
            stds = [0.229, 0.224, 0.225]
        if self.use_input_norm:
            mean = torch.Tensor(means).view(1, len(means), 1, 1)  # 1, 3, 1, 1
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor(stds).view(1, len(stds), 1, 1)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)

        params = {'num_classes': 1, 'in_ch': in_ch}
        if pretrained:
            params.update(pretrained=pretrained)
            params.update(load_fc=not drop_fc)
        print("=> creating model '{}'".format(model_name))
        if pretrained:
            self.model = EfficientNet.from_pretrained(model_name, **params)
        else:
            self.model = EfficientNet.from_name(model_name, override_params=params)

    def forward(self, x):
        x = F.interpolate(x, (self.image_size, self.image_size))

        if self.use_input_norm:
            x = (x - self.mean) / self.std

        out = self.model(x)
        return out


class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, model_name, in_ch, pretrained='imagenet', device=torch.device('cpu'), use_input_norm=True):
        super(EfficientNetFeatureExtractor, self).__init__()
        self.model_name = model_name
        self.use_input_norm = use_input_norm

        grayscale = 'gray' in model_name

        if grayscale:
            means = [0.5]
            stds = [0.5]
        else:
            means = [0.485, 0.456, 0.406]
            stds = [0.229, 0.224, 0.225]
        if self.use_input_norm:
            mean = torch.Tensor(means).view(1, len(means), 1, 1).to(device)  # 1, 3, 1, 1
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor(stds).view(1, len(stds), 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        num_classes = 1
        self.num_classes = num_classes
        self.image_size = EfficientNet.get_image_size(model_name)  # 224

        self.model = EfficientNet.from_pretrained(model_name, num_classes=num_classes, in_channels=in_ch,
                                                  pretrained=pretrained, load_fc=False)
        self.model.eval()
        for k, v in self.model.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        x = F.interpolate(x, (self.image_size, self.image_size), mode='bicubic')
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        outputs = self.model.extract_features(x)

        return outputs
