import torch
import torch.nn as nn
import numpy as np
import functools

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        b, c, h, w = y.size()
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss/(c*b*h*w)


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'gan' or self.gan_type == 'ragan':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss


class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp,
                                          grad_outputs=grad_outputs, create_graph=True,
                                          retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss

class PerceptualLoss(nn.Module):
    def __init__(self, crit='cb'):
        super(PerceptualLoss, self).__init__()
        if crit == 'cb':
            self.crit = CharbonnierLoss()
        elif crit == 'l1':
            self.crit = nn.L1Loss()
        elif crit == 'l2':
            self.crit = nn.MSELoss()
        else:
            raise NotImplementedError('Loss type [{:s}] not recognized.'.format(crit))

    def forward(self, outputs, targets):
        loss_list = []
        for features, target in zip(outputs, targets):
            target.detach()
            loss_list.append(self.crit(features, target))
        return sum(loss_list)/len(outputs)


class EdgeLoss(nn.Module):
    def __init__(self, crit='cb', device=torch.device('cpu')):
        super(EdgeLoss, self).__init__()
        kernel = [
            [0, 0, 1, 0, 0],
            [0, 1, 2, 1, 0],
            [1, 2, -16, 2, 1],
            [0, 1, 2, 1, 0],
            [0, 0, 1, 0, 0],
        ]
        kernel = np.array(kernel)
        kernel = torch.from_numpy(kernel).type(torch.FloatTensor)
        kernel = kernel.unsqueeze(0).unsqueeze(1).to(device)
        self.conv = functools.partial(nn.functional.conv2d, weight=kernel, padding=1)
        if crit == 'cb':
            self.crit = CharbonnierLoss()
        elif crit == 'l1':
            self.crit = nn.L1Loss()
        elif crit == 'l2':
            self.crit = nn.MSELoss()
        else:
            raise NotImplementedError('Loss type [{:s}] not recognized.'.format(crit))

    def forward(self, outputs, targets):
        out_edges = self.conv(outputs)
        target_edges = self.conv(targets)
        loss = self.crit(out_edges, target_edges)
        return loss # / torch.sum(torch.pow(log, 2))