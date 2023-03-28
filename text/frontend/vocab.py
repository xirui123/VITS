
from collections import OrderedDict
from typing import Iterable

__all__ = ["Vocab"]


class Vocab(object):
  """  Vocabulary.

  Args:
      symbols (Iterable[str]): Common symbols.
      padding_symbol (str, optional): Symbol for pad. Defaults to "<pad>".
      unk_symbol (str, optional): Symbol for unknow. Defaults to "<unk>"
      start_symbol (str, optional): Symbol for start. Defaults to "<s>"
      end_symbol (str, optional): Symbol for end. Defaults to "</s>"
  """

  def __init__(self,
               symbols: Iterable[str],
               padding_symbol="<pad>",
               unk_symbol="<unk>",
               start_symbol="<s>",
               end_symbol="</s>"):
    self.special_symbols = OrderedDict()
    for i, item in enumerate(
        [padding_symbol, unk_symbol, start_symbol, end_symbol]):
      if item:
        self.special_symbols[item] = len(self.special_symbols)

    self.padding_symbol = padding_symbol
    self.unk_symbol = unk_symbol
    self.start_symbol = start_symbol
    self.end_symbol = end_symbol

    self.stoi = OrderedDict()
    self.stoi.update(self.special_symbols)

    for i, s in enumerate(symbols):
      if s not in self.stoi:
        self.stoi[s] = len(self.stoi)
    self.itos = {v: k for k, v in self.stoi.items()}

  def __len__(self):
    return len(self.stoi)
#�ô��붨����һ����ΪVocab���࣬������ʻ��Vocab�Ĺ��캯������һ���������÷��ŵĿɵ������󣬲��ṩ������䡢δ֪���š���ʼ���źͽ������ŵ�ѡ�Vocab������������Ժͷ�����

#���ԣ�

#special_symbols: OrderedDict�����������ʻ���е�������ź����ǵ�������
#padding_symbol: �����š�
#unk_symbol: δ֪���š�
#start_symbol: ���ӵ���ʼ���š�
#end_symbol: ���ӵĽ������š�
#stoi: һ��OrderedDict���󣬽��ʻ���еĵ���ӳ�䵽��������
#itos: һ���ֵ���󣬽��ʻ���е�����ӳ�䵽�䵥�ʡ�

  @property
  def num_specials(self):
    """ The number of special symbols.
    """
    return len(self.special_symbols)

  # special tokens
  @property
  def padding_index(self):
    """ The index of padding symbol
    """
    return self.stoi.get(self.padding_symbol, -1)

  @property
  def unk_index(self):
    """The index of unknow symbol.
    """
    return self.stoi.get(self.unk_symbol, -1)

  @property
  def start_index(self):
    """The index of start symbol.
    """
    return self.stoi.get(self.start_symbol, -1)

  @property
  def end_index(self):
    """ The index of end symbol.
    """
    return self.stoi.get(self.end_symbol, -1)

  def __repr__(self):
    fmt = "Vocab(size: {},\nstoi:\n{})"
    return fmt.format(len(self), self.stoi)

  def __str__(self):
    return self.__repr__()

  def lookup(self, symbol):
    """ The index that symbol correspond.
    """
    return self.stoi[symbol]

  def reverse(self, index):
    """ The symbol thar index cottespond.
    """
    return self.itos[index]

  def add_symbol(self, symbol):
    """ Add a new symbol in vocab.
    """
    if symbol in self.stoi:
      return
    N = len(self.stoi)
    self.stoi[symbol] = N
    self.itos[N] = symbol

  def add_symbols(self, symbols):
    """ Add multiple symbols in vocab.
    """
    for symbol in symbols:
      self.add_symbol(symbol)
