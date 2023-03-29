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

#�ô��붨����һ��Encoder�࣬�̳���nn.Module��
#����ʵ����һ������Transformer Encoderģ�ͣ�������Ȼ���Դ����е����н�ģ����

#�ڳ�ʼ�������У�������hidden_channels�������С����filter_channels���������˲�����С����
#n_heads����ͷע������ͷ������n_layers���������Ĳ�������kernel_size������˴�С����
#p_dropout��dropout�ĸ��ʣ���window_size��ע�������ڴ�С���Ȳ�����ͬʱ����ʼ����һ��Dropout�㣬���MultiHeadAttention�㣬���LayerNorm��Ͷ��Feed Forward�㡣

#��forward�����У�������x��x_mask��������������x��ʾ�������У�
#x_mask��ʾ�������е����롣�ú�������ʹ���������õ���ע���������룬
#��ʹ��������������н����ڸǣ�Ȼ��ʹ�ö��ע������Dropout��LayerNorm��Feed Forward���б��롣
#����ٴ�ʹ�����뽫�����������ڸǣ����ر�������

#��forward������ʹ��������Ҫ��Ϊ�˴���䳤���С���Ϊ����Ȼ���Դ����У�������ı�ͨ���Ǳ䳤�ģ���ͬ����䳤�Ȳ�һ��
#����ˣ������ʹ�����룬��ô�ڽ��о����ע��������ʱ���̵����л���0��䣬�����ͻ�Ӱ��ģ�͵����ܡ�

#ʹ��������Ա���������⣬����������Ǹ���ģ����Щλ������ʵ�����룬��Щλ�������ġ��ھ����ע��������ʱ��
#ģ�ͻ���Ե����λ�õ���Ϣ��ֻ����ʵ��������м��㡣�����Ϳ��Ա�֤ģ�͵����ܣ������ܹ�����䳤���С�

#���㣨hidden layer����ָ�������н��������������֮��Ĳ㣬
#����Ԫ��ֱ�����ⲿ��������������ͨ��Ȩ�ؾ���ͼ�������������ݽ��б任�ʹ���
#��ȡ���߲�εĳ���������

#����㣨convolutional layer����ָ��������磨CNN���е�һ�ֲ�����
#������������ͨ�����ɶ��ͨ����ɵ�����ͼ��ͨ����������ͳػ��������������ݽ���������ȡ���²�����
#���ɾ��оֲ������Ժ�λ����Ϣ�ĸ߲��������

#�˲�����filter����ָ����������еľ���ˣ����Сͨ������������С��ͨ�����������ݵľ������
#�����������ݵ�ÿ��λ�ý���������ȡ����Ϣ�ۺϣ������������ͼ��

#ע����ͷ����number of attention heads����ָ��ע�������ƣ�attention mechanism���У�
#���������ݷָ�����ɸ����֣���ÿ��������ʹ�ö�����ע����������ƣ���󽫲�ͬ���ֵ�ע�������������ϣ�
#�õ����յ����������

#����ˣ�convolution kernel����ָ��������е�һ���˲��������С����״ͨ���ǹ̶��ģ�
#ͨ�����������ݵľ����������ȡ�������ݵ�ĳ�������������������ͼ��

#ע�������ڣ�attention window����ָ��ע���������У�
#���������ݰ��մ��ڴ�С���зָ��ÿ�������ڵ����ݽ��ж�����ע�����������ϣ�
#��ʵ�ֲ�ͬ����֮�����Ϣ������������ȡ��ע�������ڿ������ڼ��ټ���ͼ���ע���������в�����������
#ע����������һ�ֻ��ƣ�����������������ȷ��ÿ��Ԫ�ص������Ҫ�ԣ�����������Ҫ�Զ����ǽ��м�Ȩƽ����

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

#����һ�� Transformer �е� Decoder �࣬���̳��� nn.Module �ࡣ
#�������˶���������㣬ÿ���㶼�ɶ�ͷע�������ơ�ȫ����ǰ������Ͳв�������ɡ�

#forward ��������Ϊ self-attention �� encoder-decoder attention �ֱ𴴽����롣
#���� self-attention ������ʹ�� commons.subsequent_mask �������ɣ��������ƽ�������ǰʱ�䲽֮���λ����Ϣ��
#encoder-decoder attention ������ʹ����������ͱ���������ĳ˻����ɣ������ڱα���������䲿�֡�
#Ȼ�󣬽����� x ������������ x_mask�����ڽ���䲿�ֵ�ֵ��Ϊ 0������������ÿ��������������ִ�����²�����

#self-attention��ִ�ж�ͷע�������㣬���ά��Ϊ (batch_size, seq_len, hidden_channels)��
#�в����Ӻ͹�һ������ self-attention ����������� x ��Ӳ����� Layer Normalization�����ά����Ϊ (batch_size, seq_len, hidden_channels)��
#encoder-decoder attention��ִ�ж�ͷע�������㣬���ά��Ϊ (batch_size, seq_len, hidden_channels)��
#�в����Ӻ͹�һ������ encoder-decoder attention ���������һ���������Ӳ����� Layer Normalization��
#���ά����Ϊ (batch_size, seq_len, hidden_channels)��
#FFN �㣺ִ�� FFN ���������ά����Ϊ (batch_size, seq_len, hidden_channels)��
#�в����Ӻ͹�һ������ FFN ����������һ���������Ӳ����� Layer Normalization��
#���ά����Ϊ (batch_size, seq_len, hidden_channels)��


#�в�������һ�����������й㷺ʹ�õļ�����ͨ������������������������Ϣ��������ֱ�ӿ�Խ����㡣
#�������ڽ���ݶ���ʧ���ݶȱ�ը�����⣬�����԰�����������ѧϰ��������

#��һ�������������г��õ�һ�ּ�����ͨ��������������ź�ƽ�����淶���������ݡ�
#��һ�����������ڽ���ݶ���ʧ���ݶȱ�ը�����⣬���������ȶ��Ժ������ٶȡ�

#FFN��ָ����ȫ����ǰ������㣬�����������Ա任��һ�������Լ������ɡ�
#FFN�㳣��������ע��������������ģ�͵ı������������ѧϰ��������������ͨ���в����Ӻ͹�һ��������ѵ�������ģ�����ܡ�
#�ݶ���ʧ��ָ�����練�򴫲�ʱ���ݶ�ֵ���ŷ��򴫲����������Ӷ��𽥱�С�����յ��µͲ���������ĸ����ٶȱ�ü���������ֹͣ���¡������ͻᵼ��ģ���޷�ѧϰ����Ч���������Ӷ�Ӱ��ģ�͵����ܡ�

#�ݶȱ�ը��ָ�ݶ�ֵ�ڷ��򴫲�������������󣬵������յ��ݶ�ֵ��÷ǳ���
#��������£������ĸ��¹��̾ͻ��÷ǳ����ȶ������ܵ���ģ�Ͳ�����ɢ����������

#Ϊ�˽���ݶ���ʧ���ݶȱ�ը�����⣬ͨ�������һЩ�����������ݶȵķ�Χ������Ȩ�س�ʼ����ʹ���ݶȲü���
#���߲���һЩ����ļ��������ReLU����

#�ݶ���ָ��ʧ���������ģ�Ͳ����ĵ�������ѵ��ģ��ʱ���Ż��㷨ͨ�����򴫲��㷨������ʧ��������ģ�Ͳ������ݶȣ�
#Ȼ������ݶȸ��²�������С����ʧ�������ݶȵ�ֵ��ʾ����ʧ�����ڵ�ǰ����ֵ�µı仯�ʣ����÷����ϵ�б�ʡ�
#�ݶȵĴ�С��ʾ����ʧ�����ı仯�ʣ����������ж�ģ���Ƿ�������





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

#��ͷע����������һ��ע�������Ƶı��壬��������������ֳɶ��ͷ����ÿ��ͷ��������ѧϰ��ͬ��ע�����ֲ���
#��ʹ��ģ���ܹ�ͬʱ��ע����Ĳ�ͬ���֣�������Щ��ͬ��ע�����ֲ����ں���Ϣ��
#����������ʵ��ע�������Ƶ�ͨ�÷�����ʹ�ö�ͷע������Multi-Head Attention��ģ�顣��һ�²��裺
#1.ͨ������1D����㣬������ֱ�ת��ΪQuery��Key��Value�����У�
#Query��Key������ͬ��ά�ȣ�Value�����в�ͬ��ά�ȡ����������������ά��ͨ������$d_{model}$����ģ�͵�ά�ȴ�С��

#2����Query��Key��Value�ֱ�ͨ�����ͷ��$h$�������д�������$h$��Query��Key��Value��

#3.����ÿ��ͷ������Query������Key�ĵ����Ȼ�󽫽������һ���������ӣ��õ�ע����Ȩ�أ�Attention Weights����

#4.��ÿ��ͷ��Value���м�Ȩ��ͣ��õ���ͷע����ģ��������

#5��������ͷ�����ƴ����������ͨ��һ�����Բ���б任���õ����յ������

#���Ͼ��Ƕ�ͷע�����Ļ������̣����ܹ�ʵ�ֶԲ�ͬλ�õ�����Ĳ�ͬ�̶ȵĹ�ע���Ӷ�����ģ�͵����ܡ�
#ע�������������ѧϰ���й㷺��Ӧ�ã���������Ȼ���Դ��������ڻ������롢�ı����ɵ������С�

#1D��������������е�һ�ֲ����ͣ����ڴ���һά���������ݣ�����ʱ�����л��ı���
#��ͨ�������������Ͻ���һά�����������ȡ������������������ݡ�
#��ע���������У�Query��Key��Value�������������������ڼ���ע����Ȩ�ء�Query������ʾ��ѯ��
#Key������ʾ����Value������ʾֵ��ͨ������Query������ÿ��Key���������ƶ���ȷ��ע����Ȩ�أ�
#��ʹ����ЩȨ�ض�Value�������м�Ȩƽ�����õ�ע�������������

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
#����һ��PyTorch�е�������ģ�飬������������ģ�飺FFN��FFT��

#���ȣ���������FFNģ��Ĵ��롣����һ����������������ǰ�������磬���ڶ��������������ȡ�ͷ����Ա任��
#������������ά����������һά��ʾbatch_size���ڶ�ά��ʾ����������ά�ȣ�����ά��ʾ���г��ȡ�
#�������Ҳ��һ����ά����������������ͬ��ά�ȡ�������˵������ʵ�ֹ������£�

#��ʼ������__init__()�У�������һЩ����������������ͨ������in_channels�������ͨ������out_channels����
#����˵�ͨ������filter_channels��������˵Ĵ�С��kernel_size����dropout�ĸ��ʣ�p_dropout����
#��������ͣ�activation�����Ƿ�����������causal���������ģ���У���������ͨ������ͬ��
#��hidden_channels=filter_channels=out_channels��

#����������һά����㣨conv_1��conv_2�������ǵ�����ͨ���������ͨ����������˴�С���ɳ�����ָ����
#���У�conv_1������ͨ������in_channels�����ͨ������filter_channels��conv_2������ͨ������filter_channels��
#���ͨ������out_channels��

#ǰ�򴫲�����forward()�У����ȶ��������ݽ���mask������Ȼ�������о����������ʹ��padding����������䡣
#����������gelu�����ھ����Ӧ��gelu�����������Ӧ��relu�����������ʹ��dropout�������򻯣�
#�ٴζԾ����Ľ������mask���������շ��ؽ����
#mask����ͨ�������ڸǻ�������������е�ĳЩ���֡����ֲ����ڴ���ɱ䳤�ȵ���������ʱ�ر����ã���������Ȼ���Դ����У�
#һ�����ӿ��ܰ�����ͬ�����ĵ��ʣ������Ҫ�Խ϶̵ľ��ӽ�����䣬�Ա����뵽�������н��д���
#ͨ��ʹ��mask����������ָʾ������������Ĳ��֣�ֻ�԰�����������Ϣ�Ĳ��ֽ��м��㣬�Ӷ���������䲿�ֶԽ����Ӱ�졣

#��������FFTģ��Ĵ��룬����һ�����ڶ�ͷע�������Ƶ�ģ�飬���ڶ�������б������ȡ������
#������������������ά����������FFNģ����ͬ��������˵������ʵ�ֹ������£�

#��ʼ������__init__()�У�������һЩ����������������ͨ������hidden_channels����
#����˵�ͨ������filter_channels����ͷ����n_heads����������n_layers��������˵Ĵ�С��kernel_size����dropout�ĸ��ʣ�p_dropout�����Ƿ�ʹ���ڽ�ƫ�ã�proximal_bias�����ڽ���ʼ����proximal_init���ȡ�
#���������ɸ���ģ�飬����LayerNorm��MultiHeadAttention��FFN�ȡ�
#���У�LayerNorm�ǹ�һ���㣬MultiHeadAttention�Ƕ�ͷע�������ƣ����ڽ�����ע�������㣬
#FFN��ǰ�������磬���ڶ��������������ȡ�ͷ����Ա任��

#���������ÿ����Ԫ�У������źŵļ�Ȩ�ͱ����ݵ�������У������������з����Ա任�����һ���µ��źţ�
#��Ϊ��һ����Ԫ�����롣����������ÿ��Ա���Ϊ���������������˷������ԣ�ʹ����������Ը��õ���ϸ��ֲ�ͬ���͵ĺ�����