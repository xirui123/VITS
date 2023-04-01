import math
import math
import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

import attentions
import commons
import modules
from commons import init_weights, get_padding
from frame_prior_network import EnergyPredictor


class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0):
    super().__init__()
    filter_channels = in_channels  # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.log_flow = modules.Log()
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.flows.append(modules.Flip())

    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2))
    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(modules.Flip())

    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

  def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x)
    x = self.pre(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
      flows = self.flows
      assert w is not None

      logdet_tot_q = 0
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = torch.split(z_q, [1, 1], 1)
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
      logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q ** 2)) * x_mask, [1, 2]) - logdet_tot_q

      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z ** 2)) * x_mask, [1, 2]) - logdet_tot
      return nll + logq  # [b]
    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
      for flow in flows:
        z = flow(z, x_mask, g=x, reverse=reverse)
      z0, z1 = torch.split(z, [1, 1], 1)
      logw = z0
      return logw
#这段代码是一个神经网络模型的实现，具体来说，它实现了一个随机时长预测器（Stochastic Duration Predictor）的类。

#构造函数（__init__）接收一些参数，包括输入通道数（in_channels）、过滤器通道数（filter_channels）、
#卷积核大小（kernel_size）、Dropout率（p_dropout）、流的数量（n_flows）和条件输入通道数（gin_channels），并在初始化函数中将这些参数保存为模型的属性。

#随机时长预测器包括两个部分：预处理器和流模型。预处理器对输入数据进行卷积和dropout操作。
#流模型实现了可逆生成流（invertible generative flow），用于对输入数据建模并预测随机时长。
#在这个实现中，使用了卷积流（ConvFlow）和逆转操作（Flip）。
#在这个随机时长预测器中，卷积和dropout操作的作用是对输入的序列数据进行特征提取和降维，以便更好地进行时长预测。
#卷积层可以从时间序列数据中提取出局部相关性的特征，而dropout层则可以随机地将部分神经元的输出设置为0，以防止过拟合现象的出现。
class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)

  def forward(self, x, x_mask, g=None):
    x = torch.detach(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)
    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask
#这是一个可逆生成流模型的实现，其中包含以下几个类：
#DurationPredictor 类：一个用于预测语音合成中音素持续时间的模块，其输入是一个音素序列，输出是一个表示持续时间的标量序列。
#该模块包含卷积层、归一化层、Dropout 层和投影层，其中投影层将卷积层的输出映射到标量值。如果提供了全局条件 g，则会将其通过条件层传入模块中。
class TextEncoder(nn.Module):
  def __init__(self,
               n_vocab,
               out_channels,
               hidden_channels,
               filter_channels,
               n_heads,
               n_layers,
               kernel_size,
               p_dropout):
    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.symbol_emb = nn.Embedding(n_vocab, hidden_channels)
    nn.init.normal_(self.symbol_emb.weight, 0.0, hidden_channels ** -0.5)

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths):
    x = self.symbol_emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
    x = torch.transpose(x, 1, -1)  # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    x = self.encoder(x * x_mask, x_mask)
    return x, x_mask


class ResidualCouplingBlock(nn.Module):
  def __init__(self,
               channels,
               hidden_channels,
               kernel_size,
               dilation_rate,
               n_layers,
               n_flows=4,
               gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(
        modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers,
                                      gin_channels=gin_channels, mean_only=True))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, reverse=reverse)
    return x
#TextEncoder 类：一个用于将文本编码为音素序列表示的模块。其输入是一个文本序列，
#输出是一个表示音素序列的张量。该模块包含嵌入层、自注意力编码器、卷积层和投影层。


class PosteriorEncoder(nn.Module):
  def __init__(self,
               in_channels,
               out_channels,
               hidden_channels,
               kernel_size,
               dilation_rate,
               n_layers,
               gin_channels=0):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask
#ResidualCouplingBlock 类：一个用于实现可逆耦合的模块，用于在流模型中实现变量之间的依赖关系。
#该模块包含多个可逆耦合层和翻转层，用于生成从当前状态到下一个状态的变换。
#其中，可逆生成流是一种能够实现数据生成和数据推理的神经网络模型，它可以实现任意维度的数据变换，并且在推理过程中可以反向传播梯度以进行优化
#在这个模型中，每个可逆耦合层都将输入数据分为两个部分，其中一个部分保持不变，另一个部分根据某种规则进行变换，然后将两个部分重新组合起来，得到下一个状态的数据。
#在生成流的过程中，这些变换可以通过反向传播梯度来更新网络参数，从而实现数据生成。在推理流的过程中，这些变换可以反向应用来推断数据的分布，从而实现数据推理。

class Generator(torch.nn.Module):
  def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
               upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
    super(Generator, self).__init__()
    self.num_kernels = len(resblock_kernel_sizes)
    self.num_upsamples = len(upsample_rates)
    self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
    resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

    self.ups = nn.ModuleList()
    for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
      self.ups.append(weight_norm(
        ConvTranspose1d(upsample_initial_channel // (2 ** i), upsample_initial_channel // (2 ** (i + 1)),
                        k, u, padding=(k - u) // 2)))

    self.resblocks = nn.ModuleList()
    for i in range(len(self.ups)):
      ch = upsample_initial_channel // (2 ** (i + 1))
      for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
        self.resblocks.append(resblock(ch, k, d))

    self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
    self.ups.apply(init_weights)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

  def forward(self, x, g=None):
    x = self.conv_pre(x)
    if g is not None:
      x = x + self.cond(g)

    for i in range(self.num_upsamples):
      x = F.leaky_relu(x, modules.LRELU_SLOPE)
      x = self.ups[i](x)
      xs = None
      for j in range(self.num_kernels):
        if xs is None:
          xs = self.resblocks[i * self.num_kernels + j](x)
        else:
          xs += self.resblocks[i * self.num_kernels + j](x)
      x = xs / self.num_kernels
    x = F.leaky_relu(x)
    x = self.conv_post(x)
    x = torch.tanh(x)

    return x

  def remove_weight_norm(self):
    print('Removing weight norm...')
    for l in self.ups:
      remove_weight_norm(l)
    for l in self.resblocks:
      l.remove_weight_norm()
#Generator是一个神经网络模型，包括多个卷积和反卷积层，用于生成音频波形。
#具体来说，Generator的输入是一个一维的噪声向量，输出是一个一维的音频波形。Generator的主要结构包括：
#预处理卷积层（conv_pre），上采样层（ups），残差块（resblocks），以及后处理卷积层（conv_post）。
#其中，预处理卷积层用于将噪声向量转化为一定形状的特征图，上采样层用于将特征图转化为更高分辨率的特征图，
#残差块用于提高特征图的质量和保持分辨率，后处理卷积层用于将特征图转化为音频波形。
#Generator的具体实现细节可以参考代码中的注释。

class DiscriminatorP(torch.nn.Module):
  def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
    super(DiscriminatorP, self).__init__()
    self.period = period
    self.use_spectral_norm = use_spectral_norm
    norm_f = weight_norm if use_spectral_norm == False else spectral_norm
    self.convs = nn.ModuleList([
      norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
      norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
      norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
      norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
      norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
    ])
    self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

  def forward(self, x):
    fmap = []

    # 1d to 2d
    b, c, t = x.shape
    if t % self.period != 0:  # pad first
      n_pad = self.period - (t % self.period)
      x = F.pad(x, (0, n_pad), "reflect")
      t = t + n_pad
    x = x.view(b, c, t // self.period, self.period)

    for l in self.convs:
      x = l(x)
      x = F.leaky_relu(x, modules.LRELU_SLOPE)
      fmap.append(x)
    x = self.conv_post(x)
    fmap.append(x)
    x = torch.flatten(x, 1, -1)

    return x, fmap
#DiscriminatorP是一个判别器模型，用于判别输入的音频波形是否真实。
#具体来说，DiscriminatorP的输入是一个一维的音频波形，输出是一个标量值，表示输入的音频波形的真实程度。
#DiscriminatorP的主要结构包括：卷积层（convs）和后处理卷积层（conv_post）。
#其中，卷积层用于将一维的音频波形转化为二维的特征图，并对特征图进行多层卷积操作，
#最终输出一个一维的特征向量。后处理卷积层用于将特征向量转化为标量值。
#DiscriminatorP的具体实现细节可以参考代码中的注释。

class DiscriminatorS(torch.nn.Module):
  def __init__(self, use_spectral_norm=False):
    super(DiscriminatorS, self).__init__()
    norm_f = weight_norm if use_spectral_norm == False else spectral_norm
    self.convs = nn.ModuleList([
      norm_f(Conv1d(1, 16, 15, 1, padding=7)),
      norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
      norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
      norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
      norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
      norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
    ])
    self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

  def forward(self, x):
    fmap = []

    for l in self.convs:
      x = l(x)
      x = F.leaky_relu(x, modules.LRELU_SLOPE)
      fmap.append(x)
    x = self.conv_post(x)
    fmap.append(x)
    x = torch.flatten(x, 1, -1)

    return x, fmap
#DiscriminatorS是用于语音合成的鉴别器，它包含了多个一维卷积层（Conv1d），
#每个卷积层后面跟着一个归一化层（weight_norm或spectral_norm）和一个泄漏整流线性单元激活函数（leaky_relu）。
#这些层逐步将音频信号从1个通道、15个时间步和7个填充元素的低分辨率特征图转换为1024个通道、5个时间步和2个填充元素的高分辨率特征图。
#最后，输出特征图通过一个额外的卷积层和归一化层进行处理，输出一个实数，表示输入音频信号是真实语音的概率。

class MultiPeriodDiscriminator(torch.nn.Module):
  def __init__(self, use_spectral_norm=False):
    super(MultiPeriodDiscriminator, self).__init__()
    periods = [2, 3, 5, 7, 11]

    discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
    discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
    self.discriminators = nn.ModuleList(discs)

  def forward(self, y, y_hat):
    y_d_rs = []
    y_d_gs = []
    fmap_rs = []
    fmap_gs = []
    for i, d in enumerate(self.discriminators):
      y_d_r, fmap_r = d(y)
      y_d_g, fmap_g = d(y_hat)
      y_d_rs.append(y_d_r)
      y_d_gs.append(y_d_g)
      fmap_rs.append(fmap_r)
      fmap_gs.append(fmap_g)

    return y_d_rs, y_d_gs, fmap_rs, fmap_gs
#MultiPeriodDiscriminator是一个多周期鉴别器，它由多个DiscriminatorS组成，
#每个DiscriminatorS针对不同的音频周期（2, 3, 5, 7, 11）进行训练。对于给定的两个音频信号y和y_hat，
#该模型返回它们在每个鉴别器中的鉴别概率，以及每个鉴别器的中间特征图。

class LengthRegulator(nn.Module):
  """Length Regulator"""

  def __init__(self, hoplen, sr):
    super(LengthRegulator, self).__init__()
    self.hoplen = hoplen
    self.sr = sr

  def LR(self, x, duration, x_lengths):
    output = list()
    mel_len = list()
    x = torch.transpose(x, 1, -1)
    frame_lengths = list()

    for batch, expand_target in zip(x, duration):
      expanded = self.expand(batch, expand_target)
      output.append(expanded)
      frame_lengths.append(expanded.shape[0])

    max_len = max(frame_lengths)
    output_padded = torch.FloatTensor(x.size(0), max_len, x.size(2))
    output_padded.zero_()
    for i in range(output_padded.size(0)):
      output_padded[i, :frame_lengths[i], :] = output[i]
    output_padded = torch.transpose(output_padded, 1, -1)

    return output_padded, torch.LongTensor(frame_lengths)

  def expand(self, batch, predicted):
    out = list()
    predicted = predicted.squeeze()
    for i, vec in enumerate(batch):
      expand_size = predicted[i].item()
      vec_expand = vec.expand(max(int(expand_size), 0), -1)
      out.append(vec_expand)

    out = torch.cat(out, 0)
    return out

  def forward(self, x, duration, x_lengths):

    output, x_lengths = self.LR(x, duration, x_lengths)
    return output, x_lengths

#LengthRegulator是一个用于将音频信号的持续时间调整为目标值的模型，
#它使用一个双向长短时记忆网络（BLSTM）来学习音频信号的持续时间和音高。
#给定一个音频信号、一个目标持续时间和一个音频信号长度，该模型返回一个新的音频信号和它的长度，
#使得新的音频信号的持续时间等于目标持续时间。
class FramePriorNet(nn.Module):
  def __init__(self,
               n_vocab,
               out_channels,
               hidden_channels,
               filter_channels,
               n_heads,
               n_layers,
               kernel_size,
               p_dropout):
    super().__init__()

    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.emb = nn.Embedding(121, hidden_channels)

    self.fft_block = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)

  def forward(self, x_frame, x_mask):
    x = x_frame
    x = self.fft_block(x * x_mask, x_mask)
    x = x.transpose(1, 2)
    return x
#FramePriorNet: This class defines the frame-level prior network that takes in a sequence of one-hot encoded phoneme 
#embeddings and predicts a sequence of continuous vectors that encode the prior distribution 
#over the predicted mel-spectrogram frames.
#这段代码定义了一个名为“FramePriorNet”的神经网络类，继承了nn.Module。这个网络的作用是生成音频帧的先验分布。
#在网络初始化时，首先定义了一个大小为121（这可能是数据集中使用的音频帧大小）的嵌入层，
#将其输出通道数设置为hidden_channels。然后，使用attentions.Encoder创建一个FFT块，将隐藏通道、滤波器通道、
#多头注意力头数、编码器层数、卷积核大小和dropout概率作为输入。该FFT块将用于生成音频帧的先验分布。
#先验分布是指在考虑某个事件的概率分布时，基于先前的知识或假设所选择的一种概率分布
#这个先验分布可以帮助模型更好地学习语音信号的统计特性，使得生成的音频更加自然、连贯、清晰，同时减少一些噪声和失真的出现。

#在前向传递函数forward中，该网络接收x_frame和x_mask作为输入。x_frame是一个包含音频帧的张量，
#x_mask是一个掩码张量，用于指示哪些元素是有效的。在前向传递过程中，将x_frame乘以x_mask，然后将其输入到FFT块中，
#然后将输出转置并返回。
class PitchPredictor(nn.Module):
  def __init__(self,
               n_vocab,
               gin_channels,
               out_channels,
               hidden_channels,
               filter_channels,
               n_heads,
               n_layers,
               kernel_size,
               p_dropout):
    super().__init__()
    self.n_vocab = n_vocab  # 音素的个数，中文和英文不同
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.pitch_net = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      6,
      kernel_size,
      p_dropout)
    self.proj_f0 = nn.Conv1d(hidden_channels, 1, 1)
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, hidden_channels, 1)

  def forward(self, x, x_mask, g=None):
    x = torch.detach(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)

    x = self.pitch_net(x * x_mask, x_mask)
    x = x * x_mask
    pred_lf0 = self.proj_f0(x).squeeze(1)
    return pred_lf0
#PitchPredictor: This class predicts the log F0 (fundamental frequency) of each frame of the mel-spectrogram,
# given the sequence of one-hot encoded phoneme embeddings.

class Projection(nn.Module):
  def __init__(self,
               hidden_channels,
               out_channels):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.out_channels = out_channels
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_mask):
    stats = self.proj(x) * x_mask
    m_p, logs_p = torch.split(stats, self.out_channels, dim=1)
    return m_p, logs_p

#投影：此类将梅尔频谱图的给定隐藏特征表示投影到高斯分布的均值和对数方差参数上，
#这些参数将用于对相应的梅尔频谱图帧进行采样。

class SynthesizerTrn(nn.Module):
  """
Synthesizer for Training
"""

  def __init__(self,
               n_vocab,
               spec_channels,
               hop_length,
               sampling_rate,
               segment_size,
               inter_channels,
               hidden_channels,
               filter_channels,
               n_heads,
               n_layers,
               kernel_size,
               p_dropout,
               resblock,
               resblock_kernel_sizes,
               resblock_dilation_sizes,
               upsample_rates,
               upsample_initial_channel,
               upsample_kernel_sizes,
               n_speakers=0,
               gin_channels=0,
               use_sdp=False,
               freeze_textencoder=False,
               freeze_decoder=False,
               **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.n_speakers = n_speakers
    self.gin_channels = gin_channels

    self.use_sdp = use_sdp

    self.enc_p = TextEncoder(n_vocab,
                             inter_channels,
                             hidden_channels,
                             filter_channels,
                             n_heads,
                             n_layers,
                             kernel_size,
                             p_dropout)
    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates,
                         upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16,
                                  gin_channels=gin_channels)
    self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)

    self.duration_predictor = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

    self.lr = LengthRegulator(hop_length, sampling_rate)
    self.frame_prior_net = FramePriorNet(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads,
                                         n_layers, kernel_size, p_dropout)
    self.pitch_predictor = PitchPredictor(n_vocab, gin_channels, inter_channels, hidden_channels,
                                          filter_channels,
                                          n_heads,
                                          n_layers,
                                          kernel_size, p_dropout)
    self.energy_predictor = EnergyPredictor(hidden_channels, gin_channels)
    self.project = Projection(hidden_channels, inter_channels)

    self.pitch_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
    self.energy_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)

    if n_speakers > 1:
      self.emb_g = nn.Embedding(n_speakers, gin_channels)
    if freeze_textencoder:
      for param in self.enc_p.named_parameters():
        param[1].requires_grad = False
    if freeze_decoder:
      for param in self.dec.named_parameters():
        param[1].requires_grad = False

  def forward(self, phonemes, phonemes_lengths, f0, energy, phndur, spec, spec_lengths, sid=None):

    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
    else:
      g = None

    # 文本编码
    x, x_mask = self.enc_p(phonemes, phonemes_lengths)

    # 时长预测
    logw_ = torch.log(phndur.detach().float() + 1).unsqueeze(1) * x_mask
    logw = self.duration_predictor(x, x_mask, g=g)
    l_loss = torch.sum((logw - logw_) ** 2, [1, 2])
    x_mask_sum = torch.sum(x_mask)
    l_length = l_loss / x_mask_sum

    # f0预测
    LF0 = (2595. * torch.log10(1. + f0 / 700.)) / 500
    pred_lf0 = self.pitch_predictor(x, x_mask, g=g)
    l_pitch = F.mse_loss(LF0, pred_lf0)
    x += self.pitch_prenet(LF0.unsqueeze(1))
    pred_f0 = (torch.pow(10, pred_lf0 * 500 / 2590) - 1) * 700

    # energy预测
    norm_energy = (energy - 60) / 36
    pred_norm_energy = self.energy_predictor(x, g)
    l_energy = F.mse_loss(norm_energy, pred_norm_energy)
    x += self.energy_prenet(norm_energy.unsqueeze(1))

    # 音素级别转换成帧级
    x_frame, x_lengths = self.lr(x, phndur, phonemes_lengths)
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x_frame.size(2)), 1).to(x.device)
    x_frame = x_frame.to(x.device)

    # 帧先验网络
    x_frame = self.frame_prior_net(x_frame, x_mask)
    x_frame = x_frame.transpose(1, 2)
    m_p, logs_p = self.project(x_frame, x_mask)

    z, m_q, logs_q, y_mask = self.enc_q(spec, spec_lengths, g=g)
    z_p = self.flow(z, y_mask, g=g)

    z_slice, ids_slice = commons.rand_slice_segments(z, spec_lengths, self.segment_size)
    o = self.dec(z_slice, g=g)
    return o, l_length, l_pitch, l_energy, ids_slice, x_mask, y_mask, (
      z, z_p, m_p, logs_p, m_q, logs_q), pred_f0, pred_norm_energy, norm_energy

  def infer(self, phonemes, phonemes_lengths,
            sid=None, noise_scale=1, max_len=None, energy_control=None, pitch_control=None, duration_control=None):
    if self.n_speakers > 0:
      g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
    else:
      g = None
    x, x_mask = self.enc_p(phonemes, phonemes_lengths)

    # 时长预测
    if isinstance(duration_control, torch.Tensor):
      duration = duration_control
    else:
      if duration_control is None:
        duration_control = 1
      logw = self.duration_predictor(x, x_mask, g=g)
      w = (torch.exp(logw) * x_mask - 1) * duration_control
      duration = torch.ceil(w)

    # f0预测
    if isinstance(pitch_control, torch.Tensor):
      LF0 = (2595. * torch.log10(1. + pitch_control / 700.)) / 500
    else:
      if pitch_control is None:
        pitch_control = 1
      LF0 = self.pitch_predictor(x, x_mask, g=g) * pitch_control
    x += self.pitch_prenet(LF0.unsqueeze(1))
    F0 = (torch.pow(10, LF0 * 500 / 2590) - 1) * 700

    # energy预测
    if isinstance(energy_control, torch.Tensor):
      norm_energy = (energy_control - 60) / 36
    else:
      if energy_control is None:
        energy_control = 1
      norm_energy = (((self.energy_predictor(x, g) * 36 + 60) * energy_control) - 60) / 36
    x += self.energy_prenet(norm_energy.unsqueeze(1))
    energy = norm_energy * 36 + 60

    # 扩帧
    x_frame, x_lengths = self.lr(x, duration, phonemes_lengths)
    x_frame = x_frame.to(x.device)
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x_frame.size(2)), 1).to(x.device)

    x_frame = self.frame_prior_net(x_frame, x_mask)
    x_frame = x_frame.transpose(1, 2)
    m_p, logs_p = self.project(x_frame, x_mask)
    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, x_mask, g=g, reverse=True)
    o = self.dec((z * x_mask)[:, :, :max_len], g=g)

    return o, x_mask, (z, z_p, m_p, logs_p), duration, F0, energy

  def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    g_src = self.emb_g(sid_src).unsqueeze(-1)
    g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
    z_p = self.flow(z, y_mask, g=g_src)
    z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
    o_hat = self.dec(z_hat * y_mask, g=g_tgt)
    return o_hat, y_mask, (z, z_p, z_hat)
#SynthesizerTrn：这是定义整个神经声码器的主要类。
#它包括一个用于文本输入的编码器网络、一个用于生成梅尔频谱图的解码器网络、
#一个用于在给定文本输入的情况下估计梅尔频谱图上的后验分布的后验编码器网络，
#以及一个用于对条件分布进行建模的基于流的模型的梅尔频谱图。
#此外，它还包括几个其他子模块，例如长度调节器、持续时间预测器和音高预测器，所有这些都用于控制生成音频的各个方面。
#以上代码是一个简单的神经网络实现，它主要包括以下几个步骤：

#初始化权重和偏置：神经网络的第一步是随机初始化权重和偏置。这些权重和偏置是用来计算神经元输出的。

#前向传播：神经网络的第二步是前向传播，也称为推断过程。在这个过程中，我们将输入数据传递给神经网络，并计算每个神经元的输出值。这是通过对每个神经元的输入进行加权求和，然后通过激活函数（如Sigmoid函数）进行处理来完成的。

#计算误差：在前向传播之后，我们计算神经网络的误差。这个误差是通过将神经网络的输出与真实值进行比较来计算的。

#反向传播：一旦我们有了误差，就可以进行反向传播。在这个过程中，我们将误差从输出层向输入层传递，并计算每个权重和偏置的梯度。这是通过使用链式法则来完成的。

#更新权重和偏置：最后，我们使用梯度下降法来更新权重和偏置，以最小化误差。这是通过将梯度乘以学习率，然后从权重和偏置中减去这个值来完成的。

#重复以上步骤多次，直到神经网络的误差达到可接受的水平或训练次数达到预定值。这样，我们就可以使用训练好的神经网络进行预测或分类任务。

#首先，通过编码器网络对输入的原始音频进行特征提取和编码，得到一个固定长度的向量表示。

#然后，生成梅尔频谱图的解码器网络将该向量表示作为条件输入，并生成相应的梅尔频谱图。

#接下来，使用后验编码器网络对该梅尔频谱图进行编码，得到另一个固定长度的向量表示。


#最后，将这两个向量表示传递给基于流的模型，用于对条件分布进行建模。具体地，该模型采用自回归结构，在每个时间步生成一个梅尔频谱图的样本，直到生成整个梅尔频谱图序列。



