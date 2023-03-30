import logging

logging.getLogger("matplotlib").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)
logging.getLogger("jieba").setLevel(logging.INFO)

import torch

import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text.cleaner import text_to_sequence


# logging.getLogger("matplotlib").setLevel(logging.INFO)
# logging.getLogger("matplotlib").setLevel(logging.INFO)
def get_text(text):
  text_norm = text_to_sequence(text)
  text_norm = torch.LongTensor(text_norm)
  return text_norm


hps = utils.get_hparams_from_file("./configs/config.json")
net_g = SynthesizerTrn(
  len(symbols),
  hps.data.filter_length // 2 + 1,
  hps.data.hop_length,
  hps.data.sampling_rate,
  hps.train.segment_size // hps.data.hop_length,
  n_speakers=hps.data.n_speakers,
  **hps.model)
_ = net_g.eval()

_ = utils.load_checkpoint("/Volumes/Extend/下载/G_76000 (1).pth", net_g, None)
text1 = "。下面给大家简单介绍一下怎么使用这个教程吧！首先我们要有魔法，才能访问到谷歌的云平台。点击连接并更改运行时类型，设置硬件加速器为G P U。然后，我们再从头到尾挨个点击每个代码块的运行标志。可能需要等待一定的时间。当我们进行到语音合成部分时，就可以更改要说的文本，并设置保存的文件名啦。"
# text2 = "。下面给大家简单介绍一下怎么使用这个教程吧！首先我们要有魔法，才能访问到谷歌的云平台。点击连接并更改运行时类型，设置硬件加速器为G P U。然后，我们再从头到尾挨个点击每个代码块的运行标志。并设置保存的文件名啦。"
stn_tst = get_text(text1)
with torch.no_grad():
  x_tst = stn_tst.unsqueeze(0)
  x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
  sid = torch.LongTensor([63])
  audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667)[0][0, 0].data.cpu().float().numpy()
# ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))
print(audio.shape[0] // 44100)
#这段代码的功能是进行语音合成，将输入的文本转换为语音。
#首先，通过导入 logging 模块，设置了三个不同的日志级别(logging level)：matplotlib、numba 和 jieba 的日志级别均为 INFO，
#这意味着只有 INFO 级别及以上的日志信息才会被记录下来。
#接下来，定义了一个名为 get_text 的函数，用于将文本转换为模型可以处理的数字形式
#在软件开发中，日志是一种记录软件运行状态和事件的重要工具。日志级别是日志信息的分类和过滤机制，用于指定记录的日志信息的重要性和优先级。
#通常有以下几个级别（从低到高）：
#DEBUG：详细的调试信息，用于诊断问题。
#INFO：一般性的信息，用于确认程序是否正常运行。
#然后，通过调用 utils 模块中的 get_hparams_from_file 函数，读取配置文件中的超参数(hyperparameters)，并将其存储在 hps 变量中。
#接着，通过 SynthesizerTrn 类创建了一个语音合成器模型 net_g，并调用 eval() 方法将其设置为评估模式。
#接着，通过调用 utils 模块中的 load_checkpoint 函数，将预训练好的模型参数加载到 net_g 模型中。
#然后，定义了一个字符串变量 text1，作为输入文本。接着，调用 get_text 函数将文本转换到语音
