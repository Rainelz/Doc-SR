import torch
from torchvision import transforms
from src.models.archs.RRDBNet_arch import RRDBNet
from PIL import Image
import numpy as np


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array, then to PIL image
    Input: 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()

    if n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    img_np = img_np.astype(out_type)
    return Image.fromarray(img_np)


config = {'in_nc': 1, 'out_nc': 1, 'nf': 64,
          'nb': 23, 'upscale': 2, 'upsample_type': 'interpolate'}

class SuperGAN():
    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = RRDBNet(**config).to(device)
        self.model.eval()
        load_net = torch.load(model_path)

        self.model.load_state_dict(load_net, strict=True)

        self.prepare = transforms.Compose([
            transforms.Grayscale(num_output_channels=config['out_nc']),
            #transforms.Resize([image_size, image_size], interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            #normalize
        ])

    def process_image(self, image):
        x = self.prepare(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(x)
        processed = tensor2img(output)
        return processed

    def free(self):
        torch.cuda.empty_cache()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(epilog='superGAN inference script')

    parser.add_argument('--model-path', help='trained model path', type=str, required=True)
    parser.add_argument('--image-path', help='image to process', type=str, default=None)
    args = parser.parse_args()

    im_path = args.image_path or '../test_img.png'
    model = SuperGAN(model_path=args.model_path)
    img = Image.open(im_path)
    output = model.process_image(img)
    output.save(im_path.replace('.png', '_processed.png'))
