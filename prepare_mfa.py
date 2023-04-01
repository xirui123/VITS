import librosa
import os
import soundfile
import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from text.cleaner import text_to_phones
from text.symbols import ja_symbols


def process_text(line):
  id_, text = line.strip().split("|")
  phones = text_to_phones(text)
  phones = [ph.replace(".", "JA") if ph in ja_symbols else ph for ph in phones]
  phones = " ".join(phones)
  return (id_, phones)


lang = "zh"
if __name__ == '__main__':
  # for spk in os.listdir("data"):
  #     if os.path.exists(f"data/{spk}/transcription_raw.txt"):
  #         os.makedirs(f"mfa_temp/wavs/{spk}",exist_ok=True)
  #         with ProcessPoolExecutor(max_workers=int(cpu_count()) // 2) as executor:
  #             lines = open(f"data/{spk}/transcription_raw.txt").readlines()
  #             futures = [executor.submit(process_text, line) for line in lines]
  #             for x in tqdm.tqdm(as_completed(futures), total=len(lines)):
  #                 id_, phones = x._result
  #                 with open(f"mfa_temp/wavs/{spk}/{id_}.txt", "w") as o:
  #                     o.write(phones+"\n")
  with ProcessPoolExecutor(max_workers=int(cpu_count()) // 2) as executor:
    for spk in os.listdir(f"data/{lang}"):
      if os.path.exists(f"data/{lang}/{spk}/transcription_raw.txt"):
        os.makedirs(f"mfa_temp/wavs/{lang}/{spk}", exist_ok=True)
        lines = open(f"data/{lang}/{spk}/transcription_raw.txt").readlines()
        futures = [executor.submit(process_text, line) for line in lines]
        for x in tqdm.tqdm(as_completed(futures), total=len(lines)):
          id_, phones = x._result
          try:
            wav, sr = librosa.load(f"data/{lang}/{spk}/wavs/{id_}.wav", sr=44100)
            soundfile.write(f"mfa_temp/wavs/{lang}/{spk}/{id_}.wav", wav, sr)
            with open(f"mfa_temp/wavs/{lang}/{spk}/{id_}.txt", "w") as o:
              o.write(phones + "\n")
          except:
            print("err:", spk, id_)
          # result = f.result()
          # o.write(result)
  # ．
  # for line in open("/Volumes/Extend/下载/jsut_ver1.1 2/basic5000/transcript_utf8.txt").readlines():
  #     id_, text = line.strip().split(":")
  #     phones = text_to_phones(f"[JA]{text}[JA]")
  #     phones = " ".join(phones)
  #     with open(f"mfa_temp/wavs/jsut/{id_}.txt", "w") as o:
  #         o.write(phones + "\n")
  print("rm -rf ./mfa_temp/temp; mfa align mfa_temp/wavs/zh mfa_temp/zh_dict.dict mfa_temp/aishell3_model.zip mfa_temp/textgrids/zh --clean --overwrite -t ./mfa_temp/temp -j 5")
  print("rm -rf ./mfa_temp/temp; mfa train mfa_temp/wavs/ja/ mfa_temp/ja_dict.dict mfa_temp/model.zip mfa_temp/textgrids/ja --clean --overwrite -t ./mfa_temp/temp -j 5")
 #这段代码主要是进行音素对齐所需的数据预处理工作，包括将原始的文本转化为音素序列，对数据进行并行处理，对每个音频文件进行音素对齐操作。
 #定义一个函数 process_text，用于将输入的一行文本进行处理，返回文件名和对应的音素序列。

#利用 ProcessPoolExecutor 对数据进行并行处理，每个子进程处理一行文本，将处理结果存入 futures 列表。

#遍历 futures 列表，对每个处理结果进行后续的音素对齐操作。其中，使用 librosa 库读取音频文件，使用 MFA 工具进行音素对齐操作，并将结果存入相应的文本文件中。

#最后输出两个命令行语句，用于进行中文和日文的音素对齐操作。