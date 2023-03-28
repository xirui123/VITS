
from text.frontend.phonectic import Phonetics

"""
A phonology system with ARPABET symbols and limited punctuations. The G2P 
conversion is done by g2p_en.

Note that g2p_en does not handle words with hypen well. So make sure the input
sentence is first normalized.
"""
from text.frontend.vocab import Vocab
from g2p_en import G2p


class ARPABET(Phonetics):
  """A phonology for English that uses ARPABET as the phoneme vocabulary.
  See http://www.speech.cs.cmu.edu/cgi-bin/cmudict for more details.
  Phoneme Example Translation
      ------- ------- -----------
      AA	odd     AA D
      AE	at	AE T
      AH	hut	HH AH T
      AO	ought	AO T
      AW	cow	K AW
      AY	hide	HH AY D
      B 	be	B IY
      CH	cheese	CH IY Z
      D 	dee	D IY
      DH	thee	DH IY
      EH	Ed	EH D
      ER	hurt	HH ER T
      EY	ate	EY T
      F 	fee	F IY
      G 	green	G R IY N
      HH	he	HH IY
      IH	it	IH T
      IY	eat	IY T
      JH	gee	JH IY
      K 	key	K IY
      L 	lee	L IY
      M 	me	M IY
      N 	knee	N IY
      NG	ping	P IH NG
      OW	oat	OW T
      OY	toy	T OY
      P 	pee	P IY
      R 	read	R IY D
      S 	sea	S IY
      SH	she	SH IY
      T 	tea	T IY
      TH	theta	TH EY T AH
      UH	hood	HH UH D
      UW	two	T UW
      V 	vee	V IY
      W 	we	W IY
      Y 	yield	Y IY L D
      Z 	zee	Z IY
      ZH	seizure	S IY ZH ER
  """
  phonemes = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER',
    'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UW', 'UH', 'V', 'W', 'Y', 'Z',
    'ZH'
  ]
  punctuations = [',', '.', '?', '!']
  symbols = phonemes + punctuations
  _stress_to_no_stress_ = {
    'AA0': 'AA',
    'AA1': 'AA',
    'AA2': 'AA',
    'AE0': 'AE',
    'AE1': 'AE',
    'AE2': 'AE',
    'AH0': 'AH',
    'AH1': 'AH',
    'AH2': 'AH',
    'AO0': 'AO',
    'AO1': 'AO',
    'AO2': 'AO',
    'AW0': 'AW',
    'AW1': 'AW',
    'AW2': 'AW',
    'AY0': 'AY',
    'AY1': 'AY',
    'AY2': 'AY',
    'EH0': 'EH',
    'EH1': 'EH',
    'EH2': 'EH',
    'ER0': 'ER',
    'ER1': 'ER',
    'ER2': 'ER',
    'EY0': 'EY',
    'EY1': 'EY',
    'EY2': 'EY',
    'IH0': 'IH',
    'IH1': 'IH',
    'IH2': 'IH',
    'IY0': 'IY',
    'IY1': 'IY',
    'IY2': 'IY',
    'OW0': 'OW',
    'OW1': 'OW',
    'OW2': 'OW',
    'OY0': 'OY',
    'OY1': 'OY',
    'OY2': 'OY',
    'UH0': 'UH',
    'UH1': 'UH',
    'UH2': 'UH',
    'UW0': 'UW',
    'UW1': 'UW',
    'UW2': 'UW'
  }

  def __init__(self):
    self.backend = G2p()
    self.vocab = Vocab(self.phonemes + self.punctuations)

  def _remove_vowels(self, phone):
    return self._stress_to_no_stress_.get(phone, phone)

  def phoneticize(self, sentence, add_start_end=False):
    """ Normalize the input text sequence and convert it into pronunciation sequence.
    Args:
        sentence (str): The input text sequence.

    Returns:
        List[str]: The list of pronunciation sequence.
    """
    phonemes = [
      self._remove_vowels(item) for item in self.backend(sentence)
    ]
    if add_start_end:
      start = self.vocab.start_symbol
      end = self.vocab.end_symbol
      phonemes = [start] + phonemes + [end]
    phonemes = [item for item in phonemes if item in self.vocab.stoi]
    return phonemes

#这段代码定义了一个类，其中包含了几个方法用于将中文文本进行音标标注。
#init 方法在类实例化时会用一个后端对象和一个包含音素和标点符号列表的词汇表对象来初始化该实例。
#_remove_vowels 方法是一个私有的辅助函数，用于将一个有声调的音素映射为其相应的无声调音素。
#phoneticize 方法接受一个输入的文本序列（字符串），并将其转换为一个音素序列。它首先使用后端对象对输入句子进行G2P转换，
#然后使用 _remove_vowels 方法从结果音素中删除元音的重音。如果 add_start_end 标志为 True，则方法会在音素序列中添加起始和结束符号。
#最后，该方法会过滤掉词汇表中没有的音素，并将结果序列作为字符串列表返回。

  def numericalize(self, phonemes):
    """ Convert pronunciation sequence into pronunciation id sequence.

    Args:
        phonemes (List[str]): The list of pronunciation sequence.

    Returns:
        List[int]: The list of pronunciation id sequence.
    """
    ids = [self.vocab.lookup(item) for item in phonemes]
    return ids
#这个函数将一个发音序列列表转换为发音 ID 序列列表。它接受一个字符串列表作为输入，表示发音序列，
#并返回一个整数列表作为输出，表示相应的发音 ID 序列。
#该函数调用了 Vocab 类的一个方法，负责将每个音素映射到唯一的 ID，
#然后将此映射应用于输入列表中的每个音素，以获取其 ID。输出是这些 ID 的列表。

  def reverse(self, ids):
    """ Reverse the list of pronunciation id sequence to a list of pronunciation sequence.

    Args:
        ids( List[int]): The list of pronunciation id sequence.

    Returns:
        List[str]:
            The list of pronunciation sequence.
    """
    return [self.vocab.reverse(i) for i in ids]
#这个函数将一个发音 ID 序列的列表反转为一个发音序列的列表。它接受一个整数列表作为输入，
#表示发音 ID 序列，并返回一个字符串列表作为输出，表示相应的发音序列。该函数调用了 Vocab 类的一个方法，
#负责将每个 ID 映射回其相应的音素，并将此映射应用于输入列表中的每个 ID，以获取其相应的音素。输出是这些音素的列表。

  def __call__(self, sentence, add_start_end=False):
    """ Convert the input text sequence into pronunciation id sequence.

    Args:
        sentence (str): The input text sequence.

    Returns:
        List[str]: The list of pronunciation id sequence.
    """
    return self.numericalize(
      self.phoneticize(sentence, add_start_end=add_start_end))
#该代码定义了一个名为 __call__ 的函数，用于将输入的文本序列转换为对应的拼音ID序列。
#这个函数调用了 numericalize 和 phoneticize 函数。如果 add_start_end 参数为 True，
#则在开始和结束处添加起始和结束符号。
  @property
  def vocab_size(self):
    """ Vocab size.
    """
    # 47 = 39 phones + 4 punctuations + 4 special tokens
    return len(self.vocab)
#这段代码定义了一个属性 vocab_size，它是只读的，不能被修改。该属性返回了 vocab 的长度，
#即单词表的大小。在这个实现中，单词表大小为 47，其中包括 39 个音素、4 个标点符号和 4 个特殊符号。

class ARPABETWithStress(Phonetics):
  phonemes = [
    'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
    'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D',
    'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2',
    'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R',
    'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V',
    'W', 'Y', 'Z', 'ZH'
  ]
  punctuations = [',', '.', '?', '!']
  symbols = phonemes + punctuations
#这段代码定义了一个类 ARPABETWithStress，
#继承了 Phonetics 类。
#它定义了三个属性：phonemes、punctuations 和 symbols。

#phonemes 是一个包含 ARPABET 音素的列表，其中包括三种不同的重音标记。ARPABET 是一种用于表示英语发音的音素集合。

#punctuations 是一个包含常用英文标点符号的列表。

#symbols 是一个包含 phonemes 和 punctuations 的合并列表。

#该类的作用是用于音素的转换和处理。
  def __init__(self):
    self.backend = G2p()
    self.vocab = Vocab(self.phonemes + self.punctuations)

  def phoneticize(self, sentence, add_start_end=False):
    """ Normalize the input text sequence and convert it into pronunciation sequence.

    Args:
        sentence (str): The input text sequence.

    Returns:
        List[str]: The list of pronunciation sequence.
    """
    phonemes = self.backend(sentence)
    if add_start_end:
      start = self.vocab.start_symbol
      end = self.vocab.end_symbol
      phonemes = [start] + phonemes + [end]
    phonemes = [item for item in phonemes if item in self.vocab.stoi]
    return phonemes

  def numericalize(self, phonemes):
    """ Convert pronunciation sequence into pronunciation id sequence.

    Args:
        phonemes (List[str]): The list of pronunciation sequence.

    Returns:
        List[int]: The list of pronunciation id sequence.
    """
    ids = [self.vocab.lookup(item) for item in phonemes]
    return ids
#这段代码定义了一个名为numericalize的函数，用于将发音序列转换为发音ID序列。

#函数有一个输入参数phonemes，它是一个由字符串构成的列表，
#表示发音序列。函数首先定义一个空的列表ids，然后对于phonemes中的每一个元素，
#调用self.vocab.lookup(item)将其转换为对应的ID，并将结果添加到ids中。

#最后，函数返回ids，这是一个整数类型的列表，表示发音ID序列。
  def reverse(self, ids):
    """ Reverse the list of pronunciation id sequence to a list of pronunciation sequence.
    Args:
        ids (List[int]): The list of pronunciation id sequence.

    Returns:
        List[str]: The list of pronunciation sequence.
    """
    return [self.vocab.reverse(i) for i in ids]

  def __call__(self, sentence, add_start_end=False):
    """ Convert the input text sequence into pronunciation id sequence.
    Args:
        sentence (str): The input text sequence.

    Returns:
        List[str]: The list of pronunciation id sequence.
    """
    return self.numericalize(
      self.phoneticize(sentence, add_start_end=add_start_end))

  @property
  def vocab_size(self):
    """ Vocab size.
    """
    # 77 = 69 phones + 4 punctuations + 4 special tokens
    return len(self.vocab)


#The seventh line imports all functions and classes from the module "zh_normalization" using a relative import.
#Together, these imports make available a set of functions and classes from within the same package, 
#which can be used by the current module to perform various text processing tasks related to Chinese language, 
#such as lexicon generation, text normalization, punctuation handling, tone sandhi, and vocabulary analysis.
#此代码从同一包中导入多个模块。
#第一行使用相对导入从模块“generate_lexicon”导入所有函数和类。 “generate_lexicon”前面的点（“.”）表示它与当前模块位于同一个包中。
#第二行使用相对导入从模块“normalizer”导入所有函数和类。

#第三行使用“#”字符注释掉。它似乎正在导入一个名为“phonectic”的模块，但该行已被注释掉，因此不会执行。

#第四行使用相对导入从模块“punctuation”导入所有函数和类。

#第五行使用相对导入从模块“tone_sandhi”导入所有函数和类。

#第六行使用相对导入从模块“vocab”导入所有函数和类。

#第七行使用相对导入从模块“zh_normalization”导入所有函数和类。

#这些导入一起使同一包中的一组函数和类可用，当前模块可以使用它们来执行与中文相关的各种文本处理任务，
#例如词典生成、文本规范化、标点符号处理、变调, 和词汇分析。