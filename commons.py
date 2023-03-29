import math
import torch
from torch.nn import functional as F


def init_weights(m, mean=0.0, std=0.01):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
  return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape


def intersperse(lst, item):
  result = [item] * (len(lst) * 2 + 1)
  result[1::2] = lst
  return result


def kl_divergence(m_p, logs_p, m_q, logs_q):
  """KL(P||Q)"""
  kl = (logs_q - logs_p) - 0.5
  kl += 0.5 * (torch.exp(2. * logs_p) + ((m_p - m_q) ** 2)) * torch.exp(-2. * logs_q)
  return kl


def rand_gumbel(shape):
  """Sample from the Gumbel distribution, protect from overflows."""
  uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
  return -torch.log(-torch.log(uniform_samples))


def rand_gumbel_like(x):
  g = rand_gumbel(x.size()).to(dtype=x.dtype, device=x.device)
  return g


def slice_segments(x, ids_str, segment_size=4):
  ret = torch.zeros_like(x[:, :, :segment_size])
  for i in range(x.size(0)):
    idx_str = ids_str[i]
    idx_end = idx_str + segment_size
    ret[i] = x[i, :, idx_str:idx_end]
  return ret


def rand_slice_segments(x, x_lengths=None, segment_size=4):
  b, d, t = x.size()
  if x_lengths is None:
    x_lengths = t
  ids_str_max = x_lengths - segment_size + 1
  ids_str = (torch.rand([b]).to(device=x.device) * ids_str_max).to(dtype=torch.long)
  ret = slice_segments(x, ids_str, segment_size)
  return ret, ids_str


def get_timing_signal_1d(
    length, channels, min_timescale=1.0, max_timescale=1.0e4):
  position = torch.arange(length, dtype=torch.float)
  num_timescales = channels // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (num_timescales - 1))
  inv_timescales = min_timescale * torch.exp(
    torch.arange(num_timescales, dtype=torch.float) * -log_timescale_increment)
  scaled_time = position.unsqueeze(0) * inv_timescales.unsqueeze(1)
  signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], 0)
  signal = F.pad(signal, [0, 0, 0, channels % 2])
  signal = signal.view(1, channels, length)
  return signal


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
  b, channels, length = x.size()
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
  return x + signal.to(dtype=x.dtype, device=x.device)


def cat_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4, axis=1):
  b, channels, length = x.size()
  signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
  return torch.cat([x, signal.to(dtype=x.dtype, device=x.device)], axis)


def subsequent_mask(length):
  mask = torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)
  return mask


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
  n_channels_int = n_channels[0]
  in_act = input_a + input_b
  t_act = torch.tanh(in_act[:, :n_channels_int, :])
  s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
  acts = t_act * s_act
  return acts


def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape


def shift_1d(x):
  x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [1, 0]]))[:, :, :-1]
  return x


def sequence_mask(length, max_length=None):
  if max_length is None:
    max_length = length.max()
  x = torch.arange(max_length, dtype=length.dtype, device=length.device)
  return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
  """
  duration: [b, 1, t_x]
  mask: [b, 1, t_y, t_x]
  """
  device = duration.device

  b, _, t_y, t_x = mask.shape
  cum_duration = torch.cumsum(duration, -1)

  cum_duration_flat = cum_duration.view(b * t_x)
  path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
  path = path.view(b, t_x, t_y)
  path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
  path = path.unsqueeze(1).transpose(2, 3) * mask
  return path


def clip_grad_value_(parameters, clip_value, norm_type=2):
  if isinstance(parameters, torch.Tensor):
    parameters = [parameters]
  parameters = list(filter(lambda p: p.grad is not None, parameters))
  norm_type = float(norm_type)
  if clip_value is not None:
    clip_value = float(clip_value)

  total_norm = 0
  for p in parameters:
    param_norm = p.grad.data.norm(norm_type)
    total_norm += param_norm.item() ** norm_type
    if clip_value is not None:
      p.grad.data.clamp_(min=-clip_value, max=clip_value)
  total_norm = total_norm ** (1. / norm_type)
  return total_norm

#这是一个PyTorch模型的辅助函数集合，包含了一些用于初始化权重、计算KL散度、生成掩码、生成路径等的函数。
#下面是一些主要的函数：

#init_weights：初始化卷积层的权重。
#卷积层是神经网络中常用的一种层，其作用是对输入数据进行卷积运算，提取特征。
#初始化卷积层的权重可以使模型更容易地学习到合适的特征，提高模型的性能。在这里，
#代码中使用的是正态分布随机初始化的方法，即给权重赋予一个平均值为0，标准差为0.01的正态分布随机数。


#get_padding：计算卷积操作的padding大小。
#卷积操作的 padding（填充）指的是在输入特征图周围填充一定数量的虚拟像素，
#以便在卷积过程中保持输入特征图的大小。，有三种类型，这里使用：通常使用 same padding，即将输入的两侧均匀地填充 0，使得卷积后输出的大小与输入大小相同。这是因为 same padding 可以保留输入的边缘信息，而且在某些情况下，比如处理图像时，这种 padding 方式可以防止图像的边缘信息丢失。
#Same Padding（same mode）：在输入特征图周围填充足够数量的零元素，以保持输出特征图与输入特征图的大小相同。这种填充大小可以计算为：
#padding_size = (kernel_size - 1) // 2，其中 kernel_size 是卷积核的大小，// 表示整除运算。

#convert_pad_shape：将padding的形状从[[a, b], [c, d]]转换为[a, b, c, d]的形式。
#常常需要使用形如[a, b, c, d]的张量来表示数据，其中a代表batch size，b代表通道数，c和d代表图像尺寸。
#而在进行卷积操作时，需要将输入的图像进行padding，以避免卷积核“掉出”输入张量的边界，导致输出张量的尺寸缩小。

#kl_divergence：计算KL散度，用于衡量两个概率分布的差异。
#是信息论中用来衡量两个概率分布之间差异的一种度量。在深度学习中，KL散度通常被用来度量两个概率分布之间的距离，它越小表示这两个分布越相似。

#rand_gumbel和rand_gumbel_like：从Gumbel分布中采样。
#slice_segments和rand_slice_segments：将一个张量按照指定的大小进行切片，并返回切片后的结果。
#get_timing_signal_1d、add_timing_signal_1d和cat_timing_signal_1d：生成时序信号，用于在模型中考虑时间序列信息。
#时序信号的主要特点是信号的取值随时间而变化，其反映了某个系统在一段时间内的变化过程

#subsequent_mask：生成一个掩码，用于在self-attention中屏蔽未来时刻的信息。
#在自回归模型中，当前时刻的输出往往会被用作下一时刻的输入。在这种情况下，
#当前时刻的输出直接依赖于下一时刻的输入。如果未来时刻的信息泄漏到当前时刻，
#就会导致模型在训练和推理时产生错误的预测结果。因此，在这种情况下，需要屏蔽未来时刻的信息，
#以避免未来时刻的输入影响当前时刻的输出。这种屏蔽可以通过在self-attention中生成掩码来实现。
#生成掩码是为了在计算注意力权重时，将未来时刻的信息屏蔽掉，即将未来时刻对当前时刻的注意力权重设为0，

#fused_add_tanh_sigmoid_multiply：对两个张量进行操作，返回$\tanh(a+b)\cdot\sigma(c+d)$。
#shift_1d：将一个张量向右移动一位，并在左侧填充0。
#sequence_mask：生成一个掩码，用于在交叉注意力中屏蔽填充位置。
#generate_path：根据时长和掩码生成路径。

#