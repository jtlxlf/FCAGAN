import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import spectral_norm
from torch.optim import lr_scheduler
from torch.autograd import Variable
import functools
import numpy as np
from sklearn.decomposition import PCA
from torchvision import transforms

class Identity(nn.Module):
    def forward(self, x):
        return x

def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=True)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=True)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=True)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

def get_norm_layer(norm_type='instance'):

    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):

    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net
def  define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'FCA':
        net = FCA(ngf=ngf, use_dropout=use_dropout, n_blocks=6)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], use_spectral_norm=False):
    norm_layer = get_norm_layer(norm_type=norm)
    if netD == 'basic':  # default PatchGAN classifier Receptive Field = 70
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'basic_64_c':  # default PatchGAN classifier Receptive Field = 34
        if use_spectral_norm:
            net = NLayerDiscriminatorSstyle(input_nc, ndf, n_layers=2, norm_layer=norm_layer)
        else:
            net = NLayerDiscriminator(input_nc, ndf, n_layers=2, norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)

class NLayerDiscriminatorSstyle(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminatorSstyle, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                        spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                        norm_layer(ndf * nf_mult),
                        nn.LeakyReLU(0.2, True)
                        ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
                    spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=3, stride=1, padding=padw,bias=use_bias)),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)
                    ]

        self.nor = spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))
        self.model2 = nn.Sequential(*sequence)
        self.downsample2= nn.Sequential(nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=3, stride=2, padding=1),
                                        nn.BatchNorm2d(ndf * nf_mult),
                                        nn.ReLU ( True)
                                        )

        self.fc2 = nn.Linear(4096, 3)
        self.softmax2 = nn.Softmax(dim=1)
    def forward(self, input):
        """Standard forward."""
        B, K, _, _ = input.shape
        im1 = self.model2(input)
        nor1 =self.nor(im1)
        im2 =  self.downsample2(im1)
        nor2 = self.nor(im2)
        im3 = self.downsample2(im2)
        nor3 =self.nor(im3)
        style_features = torch.mean(im3.view(B, 256, 4, 4), dim=1)
        style_features = im3.view(B, -1)
        st = self.fc2(style_features)
        weight = self.softmax2(st.detach())
        return [ weight,nor1,nor2,nor3]
      class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'ploy':
            self.loss = nn.CrossEntropyLoss

        elif gan_mode in ['wgangp', 'hinge','hinge2']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real , train_gen=False):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'smoth——hinge2':
            if train_gen:
                loss1 = -prediction[1].mean()
                loss2 = -prediction[2].mean()
                loss3 = -prediction[3].mean()
                weight = prediction[0].mean(dim=0)
                loss = weight[0]*loss1  +  weight[1]*loss2  + weight[2]*loss3
                # loss = loss1 * prediction[0].narrow(1, 0, 1) + loss2 * prediction[0].narrow(1, 1, 1) + loss3 * \
                #        prediction[0].narrow(1, 2, 1)
            else:
                if target_is_real:
                    loss1 = 0.5 * torch.pow(torch.nn.ReLU()(1.0 - prediction[1]), 2).mean()
                    loss2 = 0.5 * torch.pow(torch.nn.ReLU()(1.0 - prediction[2]), 2).mean()
                    loss3 = 0.5 * torch.pow(torch.nn.ReLU()(1.0 - prediction[3]), 2).mean()
                    weight = prediction[0].mean(dim=0)
                    loss = weight[0] * loss1 + weight[1] * loss2 + weight[2] * loss3
                    # loss = (loss1  + loss2 + loss3)/3
                    # loss = loss1 * prediction[0].narrow(1, 0, 1) + loss2 * prediction[0].narrow(1, 1, 1) + loss3 * \
                    #        prediction[0].narrow(1, 2, 1)

                else:
                    loss1 = 0.5 * torch.pow(torch.nn.ReLU()(1.0 + prediction[1]), 2).mean()
                    loss2 = 0.5 * torch.pow(torch.nn.ReLU()(1.0 + prediction[2]), 2).mean()
                    loss3 = 0.5 * torch.pow(torch.nn.ReLU()(1.0 + prediction[3]), 2).mean()
                    loss = loss1 * prediction[0].narrow(1, 0, 1) + loss2 * prediction[0].narrow(1, 1, 1) + loss3 * \
                           prediction[0].narrow(1, 2, 1)

                    #loss = (loss1  + loss2 + loss3)/3
                    weight = prediction[0].mean(dim=0)
                    loss = weight[0] * loss1 + weight[1] * loss2 + weight[2] * loss3
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
   
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None
            
class ResnetGenerator(nn.Module):
    

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
       
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling): # 下采样层
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        mult = 2 ** n_downsampling
        # Resnet块
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        for i in range(n_downsampling):
            # 上采样层
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
            kernel_size=3, stride=2,padding=1, output_padding=1,bias=use_bias),
            norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, inp):
        
        return self.model(torch.cat(inp, dim=1))

class ResnetBlock(nn.Module)

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
       
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
      
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class FCA_attention_module(nn.Module):
    def  __init__(self, dim):
        super().__init__()
      #FCA_Conv
        sequence_i = [,]
             
        self.model_i = nn.Sequential(*sequence_i)

        self.wtj = nn.Parameter(torch.tensor([1,1],dtype=torch.float),requires_grad=True)

    def forward(self, x):
        u = x.clone()
        B,C,W,H = x.shape
        if W == []:
            attn =self.model_i(u)
       
        else:
            print("FCA——others")
        w1 = torch.exp(self.wtj[0])/torch.sum(torch.exp(self.wtj))
        w2 = torch.exp(self.wtj[1]) / torch.sum(torch.exp(self.wtj))
        # f1 = open("adapt1.txt", 'a')
        # f2 = open("adapt2.txt", 'a')
        # s1 = str(w1)
        # s2 = str(w2)
        # f1.write(s1)
        # f2.write(s2)
        # f1.write("\n")  # 换行
        # f2.write("\n")  # 换行
        return w1 * u * attn + w2 * x
