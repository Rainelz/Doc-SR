import torch
import models.archs.SRResNet_arch as SRResNet_arch
import models.archs.discriminator_vgg_arch as SRGAN_arch
import models.archs.RRDBNet_arch as RRDBNet_arch
#import models.archs.EDVR_arch as EDVR_arch
import models.archs.discriminator_effnet as EffNet_arch

# Generator
def define_G(opt):
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    # image restoration
    if which_model == 'MSRResNet':
        netG = SRResNet_arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                       nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'])
    elif which_model == 'RRDBNet':
        upsample_type = opt_net.get('upsample_type', 'interpolate')
        netG = RRDBNet_arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'],
                                    nf=opt_net['nf'], nb=opt_net['nb'], upscale=opt_net['scale'],
                                    upsample_type=upsample_type)
    # video restoration
    elif which_model == 'EDVR':
        netG = EDVR_arch.EDVR(nf=opt_net['nf'], nframes=opt_net['nframes'],
                              groups=opt_net['groups'], front_RBs=opt_net['front_RBs'],
                              back_RBs=opt_net['back_RBs'], center=opt_net['center'],
                              predeblur=opt_net['predeblur'], HR_in=opt_net['HR_in'],
                              w_TSA=opt_net['w_TSA'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG


# Discriminator
def define_D(opt):
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']

    if which_model == 'discriminator_vgg_128':
        netD = SRGAN_arch.Discriminator_VGG_128(in_nc=opt_net['in_nc'], nf=opt_net['nf'])
    elif 'efficientnet' in which_model:
        pretrained_path = opt_net.get('pretrained', None)
        netD = EffNet_arch.Discriminator_EfficientNet(which_model, in_ch=opt_net['in_nc'],
                                                      pretrained=pretrained_path, drop_fc=True)
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))
    return netD


# Define network used for perceptual loss
def define_F(opt, use_bn=False):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    opt_net = opt.get('network_F', dict())
    use_bn = opt_net.get('use_bn', False)
    use_input_norm = opt_net.get('use_input_norm', True)
    which_model = opt_net.get('which_model_F', 'VGG')
    in_ch = opt_net.get('in_nc', 3)

    if which_model == 'VGG':
    # PyTorch pretrained VGG19-54, before ReLU.
        if use_bn:
            feature_layer = 49
        else:
            feature_layer = 34
        netF = SRGAN_arch.VGGFeatureExtractor(feature_layer=feature_layer, in_ch=in_ch, use_bn=use_bn,
                                              use_input_norm=use_input_norm, device=device)
    elif 'efficientnet' in which_model:
        pretrained = opt_net['pretrained']
        blocks = opt_net.get('perceptual_blocks', [1,3,5])
        netF = EffNet_arch.EfficientNetFeatureExtractor(which_model, in_ch=in_ch, pretrained=pretrained,
                                                        device=device, use_input_norm=use_input_norm, blocks=blocks)
    else:
        raise NotImplementedError('Feature extractor model [{:s}] not recognized'.format(which_model))
    netF.eval()  # No need to train
    return netF
