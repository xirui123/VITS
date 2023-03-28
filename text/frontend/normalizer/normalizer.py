# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import unicodedata
from builtins import str as unicode

from text.frontend.normalizer.numbers import normalize_numbers


def normalize(sentence):
  """ Normalize English text.
  """
  # preprocessing
  sentence = unicode(sentence)
  sentence = normalize_numbers(sentence)
  sentence = ''.join(
    char for char in unicodedata.normalize('NFD', sentence)
    if unicodedata.category(char) != 'Mn')  # Strip accents
  sentence = sentence.lower()
  sentence = re.sub(r"[^ a-z'.,?!\-]", "", sentence)
  sentence = sentence.replace("i.e.", "that is")
  sentence = sentence.replace("e.g.", "for example")
  return sentence


#The given code defines a function named "normalize" that takes a sentence as
# an input and performs several text normalization steps on it.
#给定的代码定义了一个名为“规范化”的函数，它将一个句子作为输入并对其执行几个文本规范化步骤。

#The first step is to convert the input sentence into a Unicode string. The second step is to 
#normalize any numbers in the sentence using the "normalize_numbers" function from the "numbers" module of the "text.frontend.normalizer" package.
#第一步是将输入的句子转换成 Unicode 字符串。
#第二步是使用“text.frontend.normalizer”包的“numbers”模块中的“normalize_numbers”
#函数对句子中的任何数字进行规范化。


#The third step is to decompose any Unicode characters in the sentence into 
#their base form, i.e., stripping any accents and diacritical marks. 
#The fourth step converts the sentence to lowercase.
#第三步是将句子中的任何 Unicode 字符分解为其基本形式，即去除所有重音符号和变音符号。第四步将句子转换为小写。


#The fifth step removes any characters from the sentence that are not
# alphabets, apostrophes, commas, periods, question marks, exclamation marks, or hyphens.
#第五步从句子中删除所有非字母、撇号、逗号、句号、问号、感叹号或连字符的字符。
#最后，该函数替换了缩写“i.e.”用“that is”和“e.g.”使用“例如”并返回规范化的句子。

#Finally, the function replaces
# the abbreviations "i.e." with "that is" and "e.g." with "for example" and returns the normalized sentence.

#The code also includes some comments explaining
# the different normalization steps and the Apache License version 2.0 under which the code is distributed.
#该代码还包括一些注释，解释了不同的规范化步骤和分发代码所依据的 Apache 许可版本 2.0。
#Unicode是一种字符编码标准，它为文本中的每个字符分配了一个唯一的数字代码，用于在计算机中存储和处理文本。
#Unicode字符集包括世界上几乎所有的书写系统，包括字母、数字、标点符号、符号、汉字、日语假名、韩文和阿拉伯文等。
