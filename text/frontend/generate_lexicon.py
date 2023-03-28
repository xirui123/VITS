"""Generate lexicon and symbols for Mandarin Chinese phonology.
The lexicon is used for Montreal Force Aligner.
Note that syllables are used as word in this lexicon. Since syllables rather 
than words are used in transcriptions produced by `reorganize_baker.py`.
We make this choice to better leverage other software for chinese text to 
pinyin tools like pypinyin. This is the convention for G2P in Chinese.
为普通话音韵学生成词典和符号。
该词典用于 Montreal Force Aligner。
请注意，音节在此词典中用作单词。由于音节而不是
比单词用于 `reorganize_baker.py` 生成的转录中。
我们做出这个选择是为了更好地利用其他中文文本软件
pypinyin等拼音工具。这是中文G2P的约定。
"""
import re
from collections import OrderedDict

INITIALS = [
  'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'zh', 'ch', 'sh',
  'r', 'z', 'c', 's', 'j', 'q', 'x'
]

FINALS = [
  'a', 'ai', 'ao', 'an', 'ang', 'e', 'er', 'ei', 'en', 'eng', 'o', 'ou',
  'ong', 'ii', 'iii', 'i', 'ia', 'iao', 'ian', 'iang', 'ie', 'io', 'iou',
  'iong', 'in', 'ing', 'u', 'ua', 'uai', 'uan', 'uang', 'uei', 'uo', 'uen',
  'ueng', 'v', 've', 'van', 'vn'
]

SPECIALS = ['sil', 'sp']


def rule(C, V, R, T):
  """Generate a syllable given the initial, the final, erhua indicator, and tone.
  Orthographical rules for pinyin are applied. (special case for y, w, ui, un, iu)

  Note that in this system, 'ü' is alway written as 'v' when appeared in phoneme, but converted to
  'u' in syllables when certain conditions are satisfied.

  'i' is distinguished when appeared in phonemes, and separated into 3 categories, 'i', 'ii' and 'iii'.
  Erhua is is possibly applied to every finals, except for finals that already ends with 'r'.
  When a syllable is impossible or does not have any characters with this pronunciation, return None
  to filter it out.
  """

  # 不可拼的音节, ii 只能和 z, c, s 拼
  if V in ["ii"] and (C not in ['z', 'c', 's']):
    return None
  # iii 只能和 zh, ch, sh, r 拼
  if V in ['iii'] and (C not in ['zh', 'ch', 'sh', 'r']):
    return None

  # 齐齿呼或者撮口呼不能和 f, g, k, h, zh, ch, sh, r, z, c, s
  if (V not in ['ii', 'iii']) and V[0] in ['i', 'v'] and (
      C in ['f', 'g', 'k', 'h', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's']):
    return None

  # 撮口呼只能和 j, q, x l, n 拼
  if V.startswith("v"):
    # v, ve 只能和 j ,q , x, n, l 拼
    if V in ['v', 've']:
      if C not in ['j', 'q', 'x', 'n', 'l', '']:
        return None
    # 其他只能和 j, q, x 拼
    else:
      if C not in ['j', 'q', 'x', '']:
        return None

  # j, q, x 只能和齐齿呼或者撮口呼拼
  if (C in ['j', 'q', 'x']) and not (
      (V not in ['ii', 'iii']) and V[0] in ['i', 'v']):
    return None

  # b, p ,m, f 不能和合口呼拼，除了 u 之外
  # bm p, m, f 不能和撮口呼拼
  if (C in ['b', 'p', 'm', 'f']) and ((V[0] in ['u', 'v'] and V != "u") or
                                      V == 'ong'):
    return None

  # ua, uai, uang 不能和 d, t, n, l, r, z, c, s 拼
  if V in ['ua', 'uai',
           'uang'] and C in ['d', 't', 'n', 'l', 'r', 'z', 'c', 's']:
    return None

  # sh 和 ong 不能拼
  if V == 'ong' and C in ['sh']:
    return None

  # o 和 gkh, zh ch sh r z c s 不能拼
  if V == "o" and C in [
    'd', 't', 'n', 'g', 'k', 'h', 'zh', 'ch', 'sh', 'r', 'z', 'c', 's'
  ]:
    return None

  # ueng 只是 weng 这个 ad-hoc 其他情况下都是 ong
  if V == 'ueng' and C != '':
    return

  # 非儿化的 er 只能单独存在
  if V == 'er' and C != '':
    return None

  if C == '':
    if V in ["i", "in", "ing"]:
      C = 'y'
    elif V == 'u':
      C = 'w'
    elif V.startswith('i') and V not in ["ii", "iii"]:
      C = 'y'
      V = V[1:]
    elif V.startswith('u'):
      C = 'w'
      V = V[1:]
    elif V.startswith('v'):
      C = 'yu'
      V = V[1:]
  else:
    if C in ['j', 'q', 'x']:
      if V.startswith('v'):
        V = re.sub('v', 'u', V)
    if V == 'iou':
      V = 'iu'
    elif V == 'uei':
      V = 'ui'
    elif V == 'uen':
      V = 'un'
  result = C + V

  # Filter  er 不能再儿化
  if result.endswith('r') and R == 'r':
    return None

  # ii and iii, change back to i
  result = re.sub(r'i+', 'i', result)

  result = result + R + T
  return result


def generate_lexicon(with_tone=False, with_erhua=False):
  """Generate lexicon for Mandarin Chinese."""
  syllables = OrderedDict()

  for C in [''] + INITIALS:
    for V in FINALS:
      for R in [''] if not with_erhua else ['', 'r']:
        for T in [''] if not with_tone else ['1', '2', '3', '4', '5']:
          result = rule(C, V, R, T)
          if result:
            syllables[result] = f'{C} {V}{R}{T}'
  return syllables
#这段代码是用来生成普通话汉语的词典的。可以选择是否需要音调和儿化音，最终返回一个有序字典，
#其中包含了所有可能的音节组合以及对应的拼音。具体来说，这个函数会循环遍历所有可能的声母、韵母、儿化音和音调的组合，
#然后通过调用一个名为"rule"的函数来计算这个组合是否有效。如果是有效的，就将这个组合和对应的拼音加入到一个有序字典中，
#并最终返回这个字典。