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
#�����Ĵ��붨����һ����Ϊ���淶�����ĺ���������һ��������Ϊ���벢����ִ�м����ı��淶�����衣

#The first step is to convert the input sentence into a Unicode string. The second step is to 
#normalize any numbers in the sentence using the "normalize_numbers" function from the "numbers" module of the "text.frontend.normalizer" package.
#��һ���ǽ�����ľ���ת���� Unicode �ַ�����
#�ڶ�����ʹ�á�text.frontend.normalizer�����ġ�numbers��ģ���еġ�normalize_numbers��
#�����Ծ����е��κ����ֽ��й淶����


#The third step is to decompose any Unicode characters in the sentence into 
#their base form, i.e., stripping any accents and diacritical marks. 
#The fourth step converts the sentence to lowercase.
#�������ǽ������е��κ� Unicode �ַ��ֽ�Ϊ�������ʽ����ȥ�������������źͱ������š����Ĳ�������ת��ΪСд��


#The fifth step removes any characters from the sentence that are not
# alphabets, apostrophes, commas, periods, question marks, exclamation marks, or hyphens.
#���岽�Ӿ�����ɾ�����з���ĸ��Ʋ�š����š���š��ʺš���̾�Ż����ַ����ַ���
#��󣬸ú����滻����д��i.e.���á�that is���͡�e.g.��ʹ�á����硱�����ع淶���ľ��ӡ�

#Finally, the function replaces
# the abbreviations "i.e." with "that is" and "e.g." with "for example" and returns the normalized sentence.

#The code also includes some comments explaining
# the different normalization steps and the Apache License version 2.0 under which the code is distributed.
#�ô��뻹����һЩע�ͣ������˲�ͬ�Ĺ淶������ͷַ����������ݵ� Apache ��ɰ汾 2.0��
#Unicode��һ���ַ������׼����Ϊ�ı��е�ÿ���ַ�������һ��Ψһ�����ִ��룬�����ڼ�����д洢�ʹ����ı���
#Unicode�ַ������������ϼ������е���дϵͳ��������ĸ�����֡������š����š����֡�������������ĺͰ������ĵȡ�
