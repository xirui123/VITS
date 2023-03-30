from collections import defaultdict

langs = ["zh", "ja"]
spk2utts = defaultdict(list)
for lang in langs:
  try:
    for line in open(f"filelists/{lang}_train.list").readlines():
      spk = line.split("|")[0]
      spk2utts[spk].append(line)
  except:
    pass
val_lines = []
train_lines = []
val_n_per_spk = 2
for spk, lines in spk2utts.items():
  val_lines += lines[-val_n_per_spk:]
  train_lines += lines[:-val_n_per_spk]

with open("filelists/train.list", "w") as f:
  for line in train_lines:
    f.write(line)
with open("filelists/val.list", "w") as f:
  for line in val_lines:
    f.write(line)

#这段代码主要是对语音识别数据进行处理，将不同语言的训练数据列表合并，并按照说话人将其划分为训练集和验证集。
#首先引入了 defaultdict 数据结构，用于创建一个字典，当访问不存在的键值对时，会自动创建一个默认值，这里设置默认值为一个空列表。
#定义了一个 langs 列表，包含需要处理的语言类型。
#创建一个名为 spk2utts 的字典，用于存储每个说话人（speaker）所说的所有句子（utterance），其初始化值为一个空列表。
#对于每个语言类型，读取其对应的训练列表文件，并遍历每一行，提取出每个句子所属的说话人信息 spk，并将该行句子添加到 spk2utts 中对应说话人的列表中。
#接下来，定义了一个 val_n_per_spk 变量，用于设置每个说话人的部分句子数量，这些句子将作为验证集。
#接着，定义了两个空列表 val_lines 和 train_lines，用于存储最终的验证集和训练集句子列表。
#遍历 spk2utts 中每个说话人及其所有句子，将其按照 val_n_per_spk 的数量划分为验证集和训练集，并将划分后的列表分别添加到 val_lines 和 train_lines 中。
#最后，将训练集和验证集的列表分别写入到 train.list 和 val.list 文件中。