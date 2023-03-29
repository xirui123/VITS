import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
from modules import LayerNorm


class Encoder(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., window_size=4, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.window_size = window_size

    self.drop = nn.Dropout(p_dropout)
    self.attn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_2 = nn.ModuleList()
    for i in range(self.n_layers):
      self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, window_size=window_size))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
      self.norm_layers_2.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask):
    attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    x = x * x_mask
    for i in range(self.n_layers):
      y = self.attn_layers[i](x, x, attn_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y)

      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_2[i](x + y)
    x = x * x_mask
    return x

#该代码定义了一个Encoder类，继承自nn.Module。
#该类实现了一个多层的Transformer Encoder模型，用于自然语言处理中的序列建模任务。

#在初始化函数中，传入了hidden_channels（隐层大小）、filter_channels（卷积层的滤波器大小）、
#n_heads（多头注意力的头数）、n_layers（编码器的层数）、kernel_size（卷积核大小）、
#p_dropout（dropout的概率）和window_size（注意力窗口大小）等参数。同时，初始化了一个Dropout层，多个MultiHeadAttention层，多个LayerNorm层和多个Feed Forward层。

#在forward函数中，传入了x和x_mask两个参数，其中x表示输入序列，
#x_mask表示输入序列的掩码。该函数首先使用掩码计算得到自注意力的掩码，
#并使用掩码对输入序列进行遮盖，然后使用多层注意力、Dropout、LayerNorm和Feed Forward进行编码。
#最后再次使用掩码将编码结果进行遮盖，返回编码结果。

#在forward函数中使用掩码主要是为了处理变长序列。因为在自然语言处理中，输入的文本通常是变长的，不同的语句长度不一样
#。因此，如果不使用掩码，那么在进行卷积或注意力计算时，短的序列会用0填充，这样就会影响模型的性能。

#使用掩码可以避免这个问题，掩码的作用是告诉模型哪些位置是真实的输入，哪些位置是填充的。在卷积和注意力计算时，
#模型会忽略掉填充位置的信息，只对真实的输入进行计算。这样就可以保证模型的性能，并且能够处理变长序列。

#隐层（hidden layer）是指神经网络中介于输入层和输出层之间的层，
#其神经元不直接与外部环境相连，而是通过权重矩阵和激活函数对输入数据进行变换和处理，
#提取出高层次的抽象特征。

#卷积层（convolutional layer）是指卷积神经网络（CNN）中的一种层类型
#，其输入数据通常是由多个通道组成的特征图，通过卷积操作和池化操作对输入数据进行特征提取和下采样，
#生成具有局部不变性和位置信息的高层次特征。

#滤波器（filter）是指卷积神经网络中的卷积核，其大小通常比输入数据小，通过与输入数据的卷积操作
#，对输入数据的每个位置进行特征提取和信息聚合，生成输出特征图。

#注意力头数（number of attention heads）是指在注意力机制（attention mechanism）中，
#将输入数据分割成若干个部分，在每个部分中使用独立的注意力计算机制，最后将不同部分的注意力结果进行组合，
#得到最终的输出特征。

#卷积核（convolution kernel）是指卷积操作中的一个滤波器，其大小和形状通常是固定的，
#通过与输入数据的卷积操作，提取输入数据的某种特征，生成输出特征图。

#注意力窗口（attention window）是指在注意力机制中，
#将输入数据按照窗口大小进行分割，对每个窗口内的数据进行独立的注意力计算和组合，
#以实现不同部分之间的信息交互和特征提取。注意力窗口可以用于加速计算和减少注意力机制中参数的数量。
#注意力计算是一种机制，用于在序列数据中确定每个元素的相对重要性，并根据其重要性对它们进行加权平均。

class Decoder(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., proximal_bias=False, proximal_init=True, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init

    self.drop = nn.Dropout(p_dropout)
    self.self_attn_layers = nn.ModuleList()
    self.norm_layers_0 = nn.ModuleList()
    self.encdec_attn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_2 = nn.ModuleList()
    for i in range(self.n_layers):
      self.self_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias, proximal_init=proximal_init))
      self.norm_layers_0.append(LayerNorm(hidden_channels))
      self.encdec_attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
      self.norm_layers_1.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
      self.norm_layers_2.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask, h, h_mask):
    """
    x: decoder input
    h: encoder output
    """
    self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
    encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
    x = x * x_mask
    for i in range(self.n_layers):
      y = self.self_attn_layers[i](x, x, self_attn_mask)
      y = self.drop(y)
      x = self.norm_layers_0[i](x + y)

      y = self.encdec_attn_layers[i](x, h, encdec_attn_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y)

      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_2[i](x + y)
    x = x * x_mask
    return x

#这是一个 Transformer 中的 Decoder 类，它继承了 nn.Module 类。
#它包含了多个解码器层，每个层都由多头注意力机制、全连接前馈网络和残差连接组成。

#forward 函数首先为 self-attention 和 encoder-decoder attention 分别创建掩码。
#其中 self-attention 的掩码使用 commons.subsequent_mask 函数生成，用于限制解码器当前时间步之后的位置信息；
#encoder-decoder attention 的掩码使用输入掩码和编码器掩码的乘积生成，用于遮蔽编码器中填充部分。
#然后，将输入 x 乘以输入掩码 x_mask，用于将填充部分的值设为 0。接下来，对每个解码器层依次执行以下操作：

#self-attention：执行多头注意力计算，输出维度为 (batch_size, seq_len, hidden_channels)。
#残差连接和归一化：将 self-attention 的输出和输入 x 相加并进行 Layer Normalization，输出维度仍为 (batch_size, seq_len, hidden_channels)。
#encoder-decoder attention：执行多头注意力计算，输出维度为 (batch_size, seq_len, hidden_channels)。
#残差连接和归一化：将 encoder-decoder attention 的输出和上一步的输出相加并进行 Layer Normalization，
#输出维度仍为 (batch_size, seq_len, hidden_channels)。
#FFN 层：执行 FFN 操作，输出维度仍为 (batch_size, seq_len, hidden_channels)。
#残差连接和归一化：将 FFN 层的输出和上一步的输出相加并进行 Layer Normalization，
#输出维度仍为 (batch_size, seq_len, hidden_channels)。


#残差连接是一种在神经网络中广泛使用的技术，通过将输入与输出相加来允许信息在网络中直接跨越多个层。
#这有助于解决梯度消失和梯度爆炸等问题，并可以帮助网络更快地学习和收敛。

#归一化是在神经网络中常用的一种技术，通过对输入进行缩放和平移来规范化输入数据。
#归一化可以有助于解决梯度消失和梯度爆炸等问题，提高网络的稳定性和收敛速度。

#FFN层指的是全连接前馈网络层，它由两个线性变换和一个非线性激活函数组成。
#FFN层常用于在自注意力机制中增加模型的表达能力，可以学习非线性特征，并通过残差连接和归一化来加速训练和提高模型性能。
#梯度消失是指在网络反向传播时，梯度值随着反向传播层数的增加而逐渐变小，最终导致低层网络参数的更新速度变得极慢，甚至停止更新。这样就会导致模型无法学习到有效的特征，从而影响模型的性能。

#梯度爆炸是指梯度值在反向传播过程中逐层增大，导致最终的梯度值变得非常大。
#这种情况下，参数的更新过程就会变得非常不稳定，可能导致模型参数发散甚至崩溃。

#为了解决梯度消失和梯度爆炸的问题，通常会采用一些技术来限制梯度的范围，例如权重初始化，使用梯度裁剪，
#或者采用一些特殊的激活函数（如ReLU）。

#梯度是指损失函数相对于模型参数的导数。在训练模型时，优化算法通过反向传播算法计算损失函数对于模型参数的梯度，
#然后根据梯度更新参数来最小化损失函数。梯度的值表示了损失函数在当前参数值下的变化率，即该方向上的斜率。
#梯度的大小表示了损失函数的变化率，可以用于判断模型是否收敛。





class MultiHeadAttention(nn.Module):
  def __init__(self, channels, out_channels, n_heads, p_dropout=0., window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
    super().__init__()
    assert channels % n_heads == 0

    self.channels = channels
    self.out_channels = out_channels
    self.n_heads = n_heads
    self.p_dropout = p_dropout
    self.window_size = window_size
    self.heads_share = heads_share
    self.block_length = block_length
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init
    self.attn = None

    self.k_channels = channels // n_heads
    self.conv_q = nn.Conv1d(channels, channels, 1)
    self.conv_k = nn.Conv1d(channels, channels, 1)
    self.conv_v = nn.Conv1d(channels, channels, 1)
    self.conv_o = nn.Conv1d(channels, out_channels, 1)
    self.drop = nn.Dropout(p_dropout)

    if window_size is not None:
      n_heads_rel = 1 if heads_share else n_heads
      rel_stddev = self.k_channels ** -0.5
      self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
      self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

    nn.init.xavier_uniform_(self.conv_q.weight)
    nn.init.xavier_uniform_(self.conv_k.weight)
    nn.init.xavier_uniform_(self.conv_v.weight)
    if proximal_init:
      with torch.no_grad():
        self.conv_k.weight.copy_(self.conv_q.weight)
        self.conv_k.bias.copy_(self.conv_q.bias)

  def forward(self, x, c, attn_mask=None):
    q = self.conv_q(x)
    k = self.conv_k(c)
    v = self.conv_v(c)

    x, self.attn = self.attention(q, k, v, mask=attn_mask)

    x = self.conv_o(x)
    return x

  def attention(self, query, key, value, mask=None):
    # reshape [b, d, t] -> [b, n_h, t, d_k]
    b, d, t_s, t_t = (*key.size(), query.size(2))
    query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
    key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
    value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

    scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
    if self.window_size is not None:
      assert t_s == t_t, "Relative attention is only available for self-attention."
      key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
      rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), key_relative_embeddings)
      scores_local = self._relative_position_to_absolute_position(rel_logits)
      scores = scores + scores_local
    if self.proximal_bias:
      assert t_s == t_t, "Proximal bias is only available for self-attention."
      scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e4)
      if self.block_length is not None:
        assert t_s == t_t, "Local attention is only available for self-attention."
        block_mask = torch.ones_like(scores).triu(-self.block_length).tril(self.block_length)
        scores = scores.masked_fill(block_mask == 0, -1e4)
    p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
    p_attn = self.drop(p_attn)
    output = torch.matmul(p_attn, value)
    if self.window_size is not None:
      relative_weights = self._absolute_position_to_relative_position(p_attn)
      value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
      output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
    output = output.transpose(2, 3).contiguous().view(b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
    return output, p_attn

  def _matmul_with_relative_values(self, x, y):
    """
    x: [b, h, l, m]
    y: [h or 1, m, d]
    ret: [b, h, l, d]
    """
    ret = torch.matmul(x, y.unsqueeze(0))
    return ret

  def _matmul_with_relative_keys(self, x, y):
    """
    x: [b, h, l, d]
    y: [h or 1, m, d]
    ret: [b, h, l, m]
    """
    ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
    return ret

  def _get_relative_embeddings(self, relative_embeddings, length):
    max_relative_position = 2 * self.window_size + 1
    # Pad first before slice to avoid using cond ops.
    pad_length = max(length - (self.window_size + 1), 0)
    slice_start_position = max((self.window_size + 1) - length, 0)
    slice_end_position = slice_start_position + 2 * length - 1
    if pad_length > 0:
      padded_relative_embeddings = F.pad(
        relative_embeddings,
        commons.convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
    else:
      padded_relative_embeddings = relative_embeddings
    used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]
    return used_relative_embeddings

  def _relative_position_to_absolute_position(self, x):
    """
    x: [b, h, l, 2*l-1]
    ret: [b, h, l, l]
    """
    batch, heads, length, _ = x.size()
    # Concat columns of pad to shift from relative to absolute indexing.
    x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

    # Concat extra elements so to add up to shape (len+1, 2*len-1).
    x_flat = x.view([batch, heads, length * 2 * length])
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))

    # Reshape and slice out the padded elements.
    x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]
    return x_final

  def _absolute_position_to_relative_position(self, x):
    """
    x: [b, h, l, l]
    ret: [b, h, l, 2*l-1]
    """
    batch, heads, length, _ = x.size()
    # padd along column
    x = F.pad(x, commons.convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
    x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])
    # add 0's in the beginning that will skew the elements after reshape
    x_flat = F.pad(x_flat, commons.convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
    x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
    return x_final

  def _attention_bias_proximal(self, length):
    """Bias for self-attention to encourage attention to close positions.
    Args:
      length: an integer scalar.
    Returns:
      a Tensor with shape [1, 1, length, length]
    """
    r = torch.arange(length, dtype=torch.float32)
    diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
    return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)

#多头注意力计算是一种注意力机制的变体，它将输入向量拆分成多个头部，每个头部都可以学习不同的注意力分布。
#这使得模型能够同时关注输入的不同部分，并在这些不同的注意力分布中融合信息。
#在神经网络中实现注意力机制的通用方法是使用多头注意力（Multi-Head Attention）模块。含一下步骤：
#1.通过三个1D卷积层，将输入分别转换为Query、Key、Value。其中，
#Query和Key具有相同的维度，Value可以有不同的维度。这三个卷积层的输出维度通常都是$d_{model}$，即模型的维度大小。

#2，将Query、Key和Value分别通过多个头（$h$个）进行处理，产生$h$个Query、Key和Value。

#3.对于每个头，计算Query和所有Key的点积，然后将结果除以一个缩放因子，得到注意力权重（Attention Weights）。

#4.对每个头的Value进行加权求和，得到多头注意力模块的输出。

#5，将所有头的输出拼接起来，并通过一个线性层进行变换，得到最终的输出。

#以上就是多头注意力的基本流程，它能够实现对不同位置的输入的不同程度的关注，从而提升模型的性能。
#注意力机制在深度学习中有广泛的应用，比如在自然语言处理中用于机器翻译、文本生成等任务中。

#1D卷积层是神经网络中的一种层类型，用于处理一维的输入数据，例如时间序列或文本。
#它通过在输入数据上进行一维卷积操作来提取特征，并产生输出数据。
#在注意力机制中，Query，Key，Value是三个向量，它们用于计算注意力权重。Query向量表示查询，
#Key向量表示键，Value向量表示值。通过计算Query向量与每个Key向量的相似度来确定注意力权重，
#再使用这些权重对Value向量进行加权平均，得到注意力输出向量。

class FFN(nn.Module):
  def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0., activation=None, causal=False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.activation = activation
    self.causal = causal

    if causal:
      self.padding = self._causal_padding
    else:
      self.padding = self._same_padding

    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
    self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
    self.drop = nn.Dropout(p_dropout)

  def forward(self, x, x_mask):
    x = self.conv_1(self.padding(x * x_mask))
    if self.activation == "gelu":
      x = x * torch.sigmoid(1.702 * x)
    else:
      x = torch.relu(x)
    x = self.drop(x)
    x = self.conv_2(self.padding(x * x_mask))
    return x * x_mask

  def _causal_padding(self, x):
    if self.kernel_size == 1:
      return x
    pad_l = self.kernel_size - 1
    pad_r = 0
    padding = [[0, 0], [0, 0], [pad_l, pad_r]]
    x = F.pad(x, commons.convert_pad_shape(padding))
    return x

  def _same_padding(self, x):
    if self.kernel_size == 1:
      return x
    pad_l = (self.kernel_size - 1) // 2
    pad_r = self.kernel_size // 2
    padding = [[0, 0], [0, 0], [pad_l, pad_r]]
    x = F.pad(x, commons.convert_pad_shape(padding))
    return x


class FFT(nn.Module):
  def __init__(self, hidden_channels, filter_channels, n_heads, n_layers=1, kernel_size=1, p_dropout=0.,
               proximal_bias=False, proximal_init=True, **kwargs):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.proximal_bias = proximal_bias
    self.proximal_init = proximal_init

    self.drop = nn.Dropout(p_dropout)
    self.self_attn_layers = nn.ModuleList()
    self.norm_layers_0 = nn.ModuleList()
    self.ffn_layers = nn.ModuleList()
    self.norm_layers_1 = nn.ModuleList()
    for i in range(self.n_layers):
      self.self_attn_layers.append(
        MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout, proximal_bias=proximal_bias,
                           proximal_init=proximal_init))
      self.norm_layers_0.append(LayerNorm(hidden_channels))
      self.ffn_layers.append(
        FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, causal=True))
      self.norm_layers_1.append(LayerNorm(hidden_channels))

  def forward(self, x, x_mask):
    """
    x: decoder input
    h: encoder output
    """
    self_attn_mask = commons.subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
    x = x * x_mask
    for i in range(self.n_layers):
      y = self.self_attn_layers[i](x, x, self_attn_mask)
      y = self.drop(y)
      x = self.norm_layers_0[i](x + y)

      y = self.ffn_layers[i](x, x_mask)
      y = self.drop(y)
      x = self.norm_layers_1[i](x + y)
    x = x * x_mask
    return x
#这是一个PyTorch中的神经网络模块，包含了两个子模块：FFN和FFT。

#首先，我们来看FFN模块的代码。它是一个包含两个卷积层的前馈神经网络，用于对输入进行特征提取和非线性变换。
#它的输入是三维的张量，第一维表示batch_size，第二维表示输入特征的维度，第三维表示序列长度。
#它的输出也是一个三维的张量，与输入相同的维度。具体来说，它的实现过程如下：

#初始化函数__init__()中，定义了一些超参数，包括输入通道数（in_channels）、输出通道数（out_channels）、
#卷积核的通道数（filter_channels）、卷积核的大小（kernel_size）、dropout的概率（p_dropout）、
#激活函数类型（activation）和是否是因果卷积（causal）。在这个模块中，输入和输出通道数相同，
#即hidden_channels=filter_channels=out_channels。

#创建了两个一维卷积层（conv_1和conv_2），它们的输入通道数、输出通道数、卷积核大小都由超参数指定。
#其中，conv_1的输入通道数是in_channels，输出通道数是filter_channels；conv_2的输入通道数是filter_channels，
#输出通道数是out_channels。

#前向传播函数forward()中，首先对输入数据进行mask操作，然后对其进行卷积操作，并使用padding函数进行填充。
#如果激活函数是gelu，则在卷积后应用gelu激活函数；否则应用relu激活函数。接着使用dropout进行正则化，
#再次对卷积后的结果进行mask操作，最终返回结果。
#mask操作通常用于遮盖或忽略输入数据中的某些部分。这种操作在处理可变长度的序列数据时特别有用，例如在自然语言处理中，
#一个句子可能包含不同数量的单词，因此需要对较短的句子进行填充，以便输入到神经网络中进行处理。
#通过使用mask操作，可以指示神经网络忽略填充的部分，只对包含有意义信息的部分进行计算，从而避免了填充部分对结果的影响。

#接下来是FFT模块的代码，它是一个基于多头注意力机制的模块，用于对输入进行编码和提取特征。
#它的输入和输出都是三维的张量，与FFN模块相同。具体来说，它的实现过程如下：

#初始化函数__init__()中，定义了一些超参数，包括隐藏通道数（hidden_channels）、
#卷积核的通道数（filter_channels）、头数（n_heads）、层数（n_layers）、卷积核的大小（kernel_size）、dropout的概率（p_dropout）、是否使用邻近偏置（proximal_bias）和邻近初始化（proximal_init）等。
#创建了若干个子模块，包括LayerNorm、MultiHeadAttention和FFN等。
#其中，LayerNorm是归一化层，MultiHeadAttention是多头注意力机制，用于进行自注意力计算，
#FFN是前馈神经网络，用于对输入进行特征提取和非线性变换。

#在神经网络的每个神经元中，输入信号的加权和被传递到激活函数中，激活函数对其进行非线性变换并输出一个新的信号，
#作为下一层神经元的输入。激活函数的作用可以被视为在神经网络中引入了非线性性，使得神经网络可以更好地拟合各种不同类型的函数。