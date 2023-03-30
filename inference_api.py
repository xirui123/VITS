import os
import threading
import torch
from flask import Flask, request, send_file
from scipy.io.wavfile import write

import utils
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols

app = Flask(__name__)
mutex = threading.Lock()


def get_text(text):
  text_norm = text_to_sequence(text + "。")
  text_norm = torch.LongTensor(text_norm)
  return text_norm


hps = utils.get_hparams_from_file("configs/new.json")
net_g = SynthesizerTrn(
  len(symbols),
  hps.data.filter_length // 2 + 1,
  hps.data.hop_length,
  hps.data.sampling_rate,
  hps.train.segment_size // hps.data.hop_length,
  n_speakers=hps.data.n_speakers,
  **hps.model)

_ = net_g.eval()

_ = utils.load_checkpoint("/Volumes/Extend/下载/G_385200.pth", net_g, None)
import time


def tts(txt):
  audioname = None
  if mutex.acquire(blocking=False):
    try:
      stn_tst = get_text(txt)
      with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        t1 = time.time()
        spk = torch.LongTensor([1])

        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, sid=spk,
                            length_scale=1)[0][0, 0].data.float().numpy()
        t2 = time.time()
        audioname = "c.wav"
        write(audioname, 44100, audio)
        os.system("ffmpeg -i c.wav -ar 22050 -y converted.wav")
        audioname = "converted.wav"

        print("推理时间：", (t2 - t1), "s")
    finally:
      mutex.release()
  return audioname


@app.route('/tts')
def text_api():
  text = request.args.get('text', '')
  audio = tts(text)
  if audio is None:
    return "服务器忙"
  return send_file(audio, as_attachment=True)


if __name__ == '__main__':
  app.run("0.0.0.0", 8080)

#这段代码是一个基于 Flask 的 Web 服务器，用于接受来自客户端的文本输入并返回语音合成的音频文件。
#代码的核心函数是 tts()，它实现了语音合成的功能，将输入的文本转化为音频文件。函数中调用了 get_text() 函数，
#该函数将文本转换成模型输入的向量格式。

#这段代码实现了基于文本的语音合成服务。具体来说，它使用了一个名为net_g的语音合成模型，将输入的文本转换为音频信号，并将音频信号输出为WAV文件。
#此外，它使用了Flask web框架，使得该服务可以通过HTTP请求访问，输入要转换的文本并返回生成的音频文件。
#在多线程环境中，多个线程可能会同时访问共享资源（例如文件、数据库等），如果不对这些共享资源进行控制，就会导致数据不一致或者其他问题。
#互斥锁（Mutex）是一种用于控制对共享资源访问的机制，它可以确保同一时间只有一个线程可以访问共享资源，其他线程需要等待该线程释放锁之后才能访问。
#在这个代码中，mutex就是一个互斥锁，它用于确保同时只有一个线程可以访问net_g模型，以避免线程之间的竞争问题。
#Flask是一个基于Python的轻量级Web应用框架