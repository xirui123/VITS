
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

#��δ��붨����һ���࣬���а����˼����������ڽ������ı����������ע��
#init ��������ʵ����ʱ����һ����˶����һ���������غͱ������б�Ĵʻ���������ʼ����ʵ����
#_remove_vowels ������һ��˽�еĸ������������ڽ�һ��������������ӳ��Ϊ����Ӧ�����������ء�
#phoneticize ��������һ��������ı����У��ַ�������������ת��Ϊһ���������С�������ʹ�ú�˶����������ӽ���G2Pת����
#Ȼ��ʹ�� _remove_vowels �����ӽ��������ɾ��Ԫ������������� add_start_end ��־Ϊ True���򷽷��������������������ʼ�ͽ������š�
#��󣬸÷�������˵��ʻ����û�е����أ��������������Ϊ�ַ����б��ء�

  def numericalize(self, phonemes):
    """ Convert pronunciation sequence into pronunciation id sequence.

    Args:
        phonemes (List[str]): The list of pronunciation sequence.

    Returns:
        List[int]: The list of pronunciation id sequence.
    """
    ids = [self.vocab.lookup(item) for item in phonemes]
    return ids
#���������һ�����������б�ת��Ϊ���� ID �����б�������һ���ַ����б���Ϊ���룬��ʾ�������У�
#������һ�������б���Ϊ�������ʾ��Ӧ�ķ��� ID ���С�
#�ú��������� Vocab ���һ������������ÿ������ӳ�䵽Ψһ�� ID��
#Ȼ�󽫴�ӳ��Ӧ���������б��е�ÿ�����أ��Ի�ȡ�� ID���������Щ ID ���б�

  def reverse(self, ids):
    """ Reverse the list of pronunciation id sequence to a list of pronunciation sequence.

    Args:
        ids( List[int]): The list of pronunciation id sequence.

    Returns:
        List[str]:
            The list of pronunciation sequence.
    """
    return [self.vocab.reverse(i) for i in ids]
#���������һ������ ID ���е��б�תΪһ���������е��б�������һ�������б���Ϊ���룬
#��ʾ���� ID ���У�������һ���ַ����б���Ϊ�������ʾ��Ӧ�ķ������С��ú��������� Vocab ���һ��������
#����ÿ�� ID ӳ�������Ӧ�����أ�������ӳ��Ӧ���������б��е�ÿ�� ID���Ի�ȡ����Ӧ�����ء��������Щ���ص��б�

  def __call__(self, sentence, add_start_end=False):
    """ Convert the input text sequence into pronunciation id sequence.

    Args:
        sentence (str): The input text sequence.

    Returns:
        List[str]: The list of pronunciation id sequence.
    """
    return self.numericalize(
      self.phoneticize(sentence, add_start_end=add_start_end))
#�ô��붨����һ����Ϊ __call__ �ĺ��������ڽ�������ı�����ת��Ϊ��Ӧ��ƴ��ID���С�
#������������� numericalize �� phoneticize ��������� add_start_end ����Ϊ True��
#���ڿ�ʼ�ͽ����������ʼ�ͽ������š�
  @property
  def vocab_size(self):
    """ Vocab size.
    """
    # 47 = 39 phones + 4 punctuations + 4 special tokens
    return len(self.vocab)
#��δ��붨����һ������ vocab_size������ֻ���ģ����ܱ��޸ġ������Է����� vocab �ĳ��ȣ�
#�����ʱ�Ĵ�С�������ʵ���У����ʱ��СΪ 47�����а��� 39 �����ء�4 �������ź� 4 ��������š�

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
#��δ��붨����һ���� ARPABETWithStress��
#�̳��� Phonetics �ࡣ
#���������������ԣ�phonemes��punctuations �� symbols��

#phonemes ��һ������ ARPABET ���ص��б����а������ֲ�ͬ��������ǡ�ARPABET ��һ�����ڱ�ʾӢ�﷢�������ؼ��ϡ�

#punctuations ��һ����������Ӣ�ı����ŵ��б�

#symbols ��һ������ phonemes �� punctuations �ĺϲ��б�

#������������������ص�ת���ʹ���
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
#��δ��붨����һ����Ϊnumericalize�ĺ��������ڽ���������ת��Ϊ����ID���С�

#������һ���������phonemes������һ�����ַ������ɵ��б�
#��ʾ�������С��������ȶ���һ���յ��б�ids��Ȼ�����phonemes�е�ÿһ��Ԫ�أ�
#����self.vocab.lookup(item)����ת��Ϊ��Ӧ��ID�����������ӵ�ids�С�

#��󣬺�������ids������һ���������͵��б���ʾ����ID���С�
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
#�˴����ͬһ���е�����ģ�顣
#��һ��ʹ����Ե����ģ�顰generate_lexicon���������к������ࡣ ��generate_lexicon��ǰ��ĵ㣨��.������ʾ���뵱ǰģ��λ��ͬһ�����С�
#�ڶ���ʹ����Ե����ģ�顰normalizer���������к������ࡣ

#������ʹ�á�#���ַ�ע�͵������ƺ����ڵ���һ����Ϊ��phonectic����ģ�飬�������ѱ�ע�͵�����˲���ִ�С�

#������ʹ����Ե����ģ�顰punctuation���������к������ࡣ

#������ʹ����Ե����ģ�顰tone_sandhi���������к������ࡣ

#������ʹ����Ե����ģ�顰vocab���������к������ࡣ

#������ʹ����Ե����ģ�顰zh_normalization���������к������ࡣ

#��Щ����һ��ʹͬһ���е�һ�麯��������ã���ǰģ�����ʹ��������ִ����������صĸ����ı���������
#����ʵ����ɡ��ı��淶���������Ŵ������, �ʹʻ������