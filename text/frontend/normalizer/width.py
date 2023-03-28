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


def full2half_width(ustr):
  half = []
  for u in ustr:
    num = ord(u)
    if num == 0x3000:  # 全角空格变半角
      num = 32
    elif 0xFF01 <= num <= 0xFF5E:
      num -= 0xfee0
    u = chr(num)
    half.append(u)
  return ''.join(half)


def half2full_width(ustr):
  full = []
  for u in ustr:
    num = ord(u)
    if num == 32:  # 半角空格变全角
      num = 0x3000
    elif 0x21 <= num <= 0x7E:
      num += 0xfee0
    u = chr(num)  # to unicode
    full.append(u)

  return ''.join(full)


#这是一段Python代码，其中定义了两个函数：
#full2half_width和half2full_width，它们的作用是将字符串中的全角字符转换为半角字符，或者将半角字符转换为全角字符。

#full2half_width函数中的for循环遍历输入字符串中的每个字符，
#判断其Unicode编码值，如果该字符是全角空格（编码值为0x3000），则将其转换为半角空格（编码值为32）。
#对于其他的全角字符，其编码值应在0xFF01到0xFF5E之间，所以将其减去0xFEE0即可得到对应的半角字符的编码值
#。最后将转换后的字符添加到列表half中，并使用join函数将列表中的字符拼接成字符串返回。