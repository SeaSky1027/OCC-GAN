import logging

import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F

def init_linear(linear):
    init.xavier_uniform_(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class SpectralNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            v = weight_mat.t() @ u
            v = v / v.norm()
            u = weight_mat @ v
            u = u / u.norm()
        sigma = u @ weight_mat @ v
        weight_sn = weight / sigma

        return weight_sn, u, sigma

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        input_size = weight.size(0)
        u = weight.new_empty(input_size).normal_()
        module.register_buffer(name, weight)
        module.register_buffer(name + '_u', u)
        module.register_buffer(name + '_sv', torch.ones(1).squeeze())

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u, sigma = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)
        setattr(module, self.name + '_sv', sigma)


def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)

    return module


def spectral_init(module, gain=1):
    init.xavier_uniform_(module.weight, gain)
    if module.bias is not None:
        module.bias.data.zero_()

    return spectral_norm(module)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, gain=2 ** 0.5):
        super().__init__()

        self.query = spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),
                                   gain=gain)
        self.key = spectral_init(nn.Conv1d(in_channel, in_channel // 8, 1),
                                 gain=gain)
        self.value = spectral_init(nn.Conv1d(in_channel, in_channel, 1),
                                   gain=gain)

        self.gamma = nn.Parameter(torch.tensor(0.0))

    def forward(self, input): # [bsz, channel, freq, time]
        shape = input.shape
        flatten = input.view(shape[0], shape[1], -1) # [bsz, channel, freq*time]
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key) # [bsz, freq*time, freq*time]
        attention_map = F.softmax(query_key, 1)
        out = torch.bmm(value, attention_map)
        out = out.view(*shape)
        out = self.gamma * out + input

        return (out, attention_map)

class ConditionalNorm(nn.Module):
    def __init__(self, in_channel, condition_dim):
        super().__init__()

        self.bn = nn.BatchNorm2d(in_channel, affine=False)
        self.linear1 = nn.Linear(condition_dim, in_channel)
        self.linear2 = nn.Linear(condition_dim, in_channel)

    def forward(self, input, condition):
        out = self.bn(input)
        gamma, beta = self.linear1(condition), self.linear2(condition)
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta

        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=[3, 3],
                 padding=1, stride=1, condition_dim=None, bn=True,
                 activation=F.relu, upsample=True, downsample=False):
        super().__init__()

        gain = 2 ** 0.5

        self.conv1 = spectral_init(nn.Conv2d(in_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=False if bn else True),
                                   gain=gain)
        self.conv2 = spectral_init(nn.Conv2d(out_channel, out_channel,
                                             kernel_size, stride, padding,
                                             bias=False if bn else True),
                                   gain=gain)

        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            self.conv_skip = spectral_init(nn.Conv2d(in_channel, out_channel,
                                                     1, 1, 0))
            self.skip_proj = True

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn
        if bn:
            self.norm1 = ConditionalNorm(in_channel, condition_dim)
            self.norm2 = ConditionalNorm(out_channel, condition_dim)

    def forward(self, input, condition=None):
        out = input

        if self.bn:
            out = self.norm1(out, condition)
        out = self.activation(out)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode='nearest')
        out = self.conv1(out)
        if self.bn:
            out = self.norm2(out, condition)
        out = self.activation(out)
        out = self.conv2(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        if self.skip_proj:
            skip = input
            if self.upsample:
                skip = F.interpolate(skip, scale_factor=2, mode='nearest')
            skip = self.conv_skip(skip)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2)

        else:
            skip = input

        return out + skip


class Generator(nn.Module):
    def __init__(self, out_channel=1, channel=128, noise_dim=128, num_classes=7, embedding_dim=128):
        super().__init__()
        self.noise_dim = noise_dim
        self.channel = channel

        self.lin_code = spectral_init(nn.Linear(noise_dim, channel * 10 * 43))
        self.conv1 = ConvBlock(channel, channel // 2, condition_dim=embedding_dim)
        self.conv2 = ConvBlock(channel // 2, channel // 2, condition_dim=embedding_dim, upsample=False)
        self.attention = SelfAttention(channel // 2)
        self.conv3 = ConvBlock(channel // 2, channel // 4, condition_dim=embedding_dim)
        self.conv4 = ConvBlock(channel // 4, channel // 4, condition_dim=embedding_dim)

        self.bn = nn.BatchNorm2d(channel // 4)
        self.colorize = spectral_init(nn.Conv2d(channel // 4, out_channel, [3, 3], padding=1))
        self.embedding = spectral_norm(nn.Embedding(num_embeddings=num_classes, embedding_dim=embedding_dim))

    def forward(self, label):
        bsz = label.shape[0]
        noise = torch.randn((bsz, self.noise_dim)).to(label.device)

        out = self.lin_code(noise)
        out = out.view(-1, self.channel, 10, 43) # [bsz, 128, 10, 43]
        condition = self.embedding(label)

        out = self.conv1(out, condition) # [bsz, 64, 20, 86]
        out = self.conv2(out, condition) # [bsz, 64, 20, 86]
        out, attention_map = self.attention(out) # [bsz, 64, 20, 86]
        out = self.conv3(out, condition) # [bsz, 32, 40, 172]
        out = self.conv4(out, condition) # [bsz, 32, 80, 344]

        out = self.bn(out) # [bsz, 32, 80, 344]
        out = F.relu(out)
        out = self.colorize(out) # [bsz, 1, 80, 344]

        return out


class Discriminator(nn.Module):
    def __init__(self, in_channel=1, channel=32, num_classes=7, embedding_dim=128):
        super().__init__()
        self.num_classes = num_classes

        def conv(in_channel, out_channel, downsample=True):
            return ConvBlock(in_channel, out_channel,
                             bn=False,
                             upsample=False, downsample=downsample)

        gain = 2 ** 0.5

        self.pre_conv = nn.Sequential(spectral_init(nn.Conv2d(in_channel, channel, 3,
                                                              padding=1),
                                                    gain=gain),
                                      nn.ReLU(),
                                      spectral_init(nn.Conv2d(channel, channel, 3,
                                                              padding=1),
                                                    gain=gain),
                                      nn.AvgPool2d(2))
        self.pre_skip = spectral_init(nn.Conv2d(in_channel, channel, 1))

        self.conv1 = conv(channel, channel * 2)
        self.conv2 = conv(channel * 2, channel * 2, downsample=False)
        self.attention = SelfAttention(channel * 2)
        self.conv3 = conv(channel * 2, channel * 4)
        self.conv4 = conv(channel * 4, channel * 4)

        self.linear = spectral_init(nn.Linear(channel * 4, 1))

        self.projection = nn.Sequential(
            spectral_init(nn.Linear(channel * 4, channel * 4)),
            nn.ReLU(),
            spectral_init(nn.Linear(channel * 4, channel * 4))
        )

        self.embedding = spectral_norm(nn.Embedding(num_embeddings=num_classes, embedding_dim=channel * 4))

    def forward(self, input, label):
        out = self.pre_conv(input) # [bsz, 32, 40, 172]
        out = out + self.pre_skip(F.avg_pool2d(input, 2)) # [bsz, 32, 40, 172]

        out = self.conv1(out) # [bsz, 64, 20, 86]
        out = self.conv2(out) # [bsz, 64, 20, 86]
        out, attention_map = self.attention(out) # [bsz, 64, 20, 86]
        out = self.conv3(out) # [bsz, 128, 10, 43]
        out = self.conv4(out) # [bsz, 128, 5, 21]

        out = F.relu(out)
        out = out.view(out.size(0), out.size(1), -1) # [bsz, 128, 105]
        out = out.sum(2) # [bsz, 128]
        adv_output = self.linear(out).squeeze(1) # [bsz, 1]

        condition = self.embedding(label) # [bsz, 128]
        prod = (out * condition).sum(1) # [bsz, 1]
        adv_output += prod

        contrastive_feature = self.projection(out) # [bsz, 128]

        return adv_output, contrastive_feature, condition

def count_parameters(module):
    num_params = sum(p.numel() for p in module.parameters())
    return num_params

if __name__ == '__main__':
    from HiFiGanWrapper import HiFiGanWrapper
    import numpy as np

    generator = Generator().eval()
    num_params = count_parameters(generator)
    print(f"Number of generator parameters: {num_params / 1000000:.2f} M")
    print()

    discriminator = Discriminator().eval()
    num_params = count_parameters(discriminator)
    print(f"Number of discriminator parameters: {num_params / 1000000:.2f} M")
    print()

    vocoder = HiFiGanWrapper(ckpt_path='./pretrained_checkpoints')
    num_params = count_parameters(vocoder.generator)
    print(f"Number of vocoder parameters: {num_params / 1000000:.2f} M")
    print()

    image = torch.randn(4, 1, 80, 344)
    labels = torch.LongTensor([0, 0, 1, 2])

    out, contrastive_feature, proxy = discriminator(image, labels)
    print('discriminator :', out.shape)
    print('contrastive_feature :', contrastive_feature.shape)
    print('proxy :', proxy.shape)
    print()

    out = generator(labels)
    print('generator :', out.shape)
    print()

    fake_sound = vocoder.generate_audio(out[0])
    fake_sound = np.concatenate((fake_sound, fake_sound[-136:]), axis=0)
    print('generated sound :', fake_sound.shape)