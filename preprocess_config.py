import json

config = json.load(open("configs/config.json"))
spk2id = {}
sid = 0
for line in open(f"filelists/train.list").readlines():
  spk = line.split("|")[0]
  if spk not in spk2id.keys():
    spk2id[spk] = sid
    sid += 1

config["data"]['spk2id'] = spk2id

with open("configs/config.json", "w") as f:
  json.dump(config, f, indent=2)

#这段代码是用来修改配置文件的，主要分为以下几个步骤：

#读取配置文件 configs/config.json 中的内容，使用 json.load() 函数将其转换为字典形式，存储在 config 变量中。

#定义一个空字典 spk2id，用来存储说话人和对应的 id。

#遍历训练数据文件 filelists/train.list 中的每一行，获取当前行的说话人 spk，如果该说话人还未出现过，则将其加入到 spk2id 字典中，将其赋予一个新的 id 值。

#将 spk2id 字典存储到 config 变量中的 data 字典的 spk2id 键下。

#将修改后的 config 变量重新写回到配置文件 configs/config.json 中，使用 json.dump() 函数将字典转换为 JSON 格式写入文件中。其中 indent=2 参数表示输出 JSON 时使用两个空格缩进。