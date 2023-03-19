import re

from text import cleaned_text_to_sequence
from text.en_frontend import en_to_phonemes
from text.ja_frontend import ja_to_phonemes
from text.mix_frontend import others_to_phonemes
from text.symbols import symbols
from text.zh_frontend import pinyin_to_phonemes
from text.zh_frontend import zh_to_phonemes

source = ["-", "--"]
target = ["sp", "sp"]
mapping = dict(zip(source, target))


def str_replace(data):
  chinaTab = ['：', '；', '，', '。', '！', '？', '【', '】', '“', '（', '）', '%', '#', '@', '&', "‘", ' ', '\n', '”', "—", "·",
              '、', '...', "―", "～"]
  englishTab = [',', ',', ',', '.', '!', '?', '[', ']', '"', '(', ')', '%', '#', '@', '&', "'", ' ', '', '"', "-", "-",
                ",", "…", ",", ","]
  for index in range(len(chinaTab)):
    if chinaTab[index] in data:
      data = data.replace(chinaTab[index], englishTab[index])
  return data


def remove_invalid_phonemes(phonemes):
  # 移除未识别的音素符号
  new_phones = []
  for ph in phonemes:
    ph = mapping.get(ph, ph)
    if ph in symbols:
      new_phones.append(ph)
    else:
      print("skip：", ph)
  return new_phones


def text_to_sequence(text):
  phones = text_to_phones(text)
  return cleaned_text_to_sequence(phones)


def text_to_phones(text: str) -> list:
  text = str_replace(text).replace("\"", '')
  # find all text blocks enclosed in [JA], [ZH], [EN], [P]
  original_text = text
  blocks = re.finditer(r'\[(JA|ZH|EN|P)\](.*?)\[\1\]', text)
  phonemes = []
  last_end = 0
  for block in blocks:
    start, end = block.span()
    # insert text not enclosed in any blocks
    remaining_text = original_text[last_end:start]
    phonemes += others_to_phonemes(remaining_text)
    last_end = end
    language = block.group(1)
    text = block.group(2)
    if language == 'P':
      phonemes += pinyin_to_phonemes(text)
    if language == 'JA':
      phonemes += ja_to_phonemes(text)
    elif language == 'ZH':
      phonemes += zh_to_phonemes(text)
    elif language == 'EN':
      phonemes += en_to_phonemes(text)
  remaining_text = original_text[last_end:]
  phonemes += others_to_phonemes(remaining_text)
  phonemes = remove_invalid_phonemes(phonemes)
  return phonemes


if __name__ == '__main__':
  test_text = "[JA]こんにちは。こんにちは\}{ll[JA]abc你好[ZH]你好[ZH][EN]Hello你好.vits![EN][JA]こんにちは。[JA]"
  text = "借还款,他只是一个纸老虎，开户行，奥大家好33啊我是Ab3s,?萨达撒abst 123、~~、、 但是、、、A B C D!"
  text = "奥大家,好33啊,こんにちは我是Ab3s,?萨达撒abst 123、~~、、 但*是、、、A B C D!"
  # text = '[JA]シシ…すご,,いじゃんシシラシャミョンありがとうえーっとシンモーレシンモーレじゃんシシラシャミョン[JA]'
  # text="大家好#要筛#，你好#也#要筛#。"
  # text = "嗯？什么东西…沉甸甸的…下午1:00，今天是2022/5/10"
  # text = "A I人工智能"
  # text = '[P]pin1 yin1 zhen1 hao3 wan2[P]扎堆儿-#'
  # text = "[JA]それより仮屋さん,例の件ですが――[JA]"
  # text = "早上好，今天是2020/10/29，最低温度是-3°C。"
  # # text = "…………"
  # print(text_to_sequence(text))
  #
  # print(time.time()-t)
  print(text_to_phones(text))
