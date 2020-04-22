import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
                                     0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class UpSampleBlock(nn.Module):
    def __init__(self, nf, scale=2):
        super(UpSampleBlock, self).__init__()

        self.scale = scale
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        out = self.act(self.conv(F.interpolate(x, scale_factor=self.scale, mode='nearest')))
        return out


class PixelShuffleBlock(nn.Module):

    def __init__(self, nf, scale=2):
        super(PixelShuffleBlock, self).__init__()
        self.upconv = nn.Conv2d(nf, nf*(scale**2), 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        out = self.act(self.conv(self.pixel_shuffle(self.upconv(x))))
        return out


class MixedShuffleBlock(nn.Module):

    def __init__(self, nf, scale=2):
        super(MixedShuffleBlock, self).__init__()
        self.pix_block = PixelShuffleBlock(nf, scale)
        self.upsample_block = UpSampleBlock(nf, scale)
        self.conv = nn.Conv2d(nf*2, nf, 3, 1, 1, bias=True)

    def forward(self, x):
        x_1 = self.pix_block(x)
        x_2 = self.upsample_block(x)
        out = self.conv(torch.cat((x_1, x_2), 1))
        return out


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, upsample_type='interpolate'):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = arch_util.make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upsamples = nn.ModuleList([])
        #### upsampling
        for _ in range(0, int(math.log(upscale, 2))):  # [1,2,4,8,...]

            if upsample_type == 'interpolate':
                self.upsamples.append(UpSampleBlock(nf))  # upsample by 2

            elif upsample_type == 'pixel_shuffle':
                self.upsamples.append(PixelShuffleBlock(nf))
            elif upsample_type == 'mixed':
                self.upsamples.append(MixedShuffleBlock(nf))
            else:
                raise NotImplementedError('Upsample type [{:s}] not recognized.'.format(upsample_type))

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk
        for upsample in self.upsamples:
            fea = upsample(fea)
        # fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        # fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
