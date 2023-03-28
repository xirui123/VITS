

__all__ = ["get_punctuations"]

EN_PUNCT = [
  " ",
  "-",
  "...",
  ",",
  ".",
  "?",
  "!",
]

CN_PUNCT = ["、", "，", "；", "：", "。", "？", "！"]


def get_punctuations(lang):
  if lang == "en":
    return EN_PUNCT
  elif lang == "cn":
    return CN_PUNCT
  else:
    raise ValueError(f"language {lang} Not supported")
#这段代码定义了一个列表__all__，其中包含了模块可供调用的函数名。
#接下来定义了两个变量EN_PUNCT和CN_PUNCT，分别存储了英文和中文标点符号的列表。
#最后定义了一个函数get_punctuations(lang)，
#该函数接收一个参数lang表示语言类型（"en"表示英文，"cn"表示中文），
#如果lang为"en"则返回英文标点符号列表，如果为"cn"则返回中文标点符号列表，
#否则抛出一个ValueError异常，表示不支持该语言类型。