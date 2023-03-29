# -*- coding: utf-8 -*-
import os
import random
import torch
import torch.utils.data

from mel_processing import spectrogram_torch
from text import cleaned_text_to_sequence
from utils import load_wav_to_torch, load_filepaths_and_text

"""Multi speaker version"""


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
  """
      1) loads audio, text pairs
      2) normalizes text and converts them to sequences of integers
      3) computes spectrograms from audio files.
  """

  def __init__(self, audiopaths_and_text, hparams):
    # 从training_files中加载的音频地址以及音频内容
    self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
    self.max_wav_value = hparams.max_wav_value
    self.sampling_rate = hparams.sampling_rate
    self.filter_length = hparams.filter_length
    self.hop_length = hparams.hop_length
    self.win_length = hparams.win_length
    self.sampling_rate = hparams.sampling_rate
    self.spk_map = hparams.spk2id

    # self.add_blank = hparams.add_blank
    # self.min_text_len = getattr(hparams, "min_text_len", 1)
    # self.max_text_len = getattr(hparams, "max_text_len", 190)

    random.seed(1234)
    random.shuffle(self.audiopaths_and_text)
    self._filter()

  def _filter(self):
    """
    Filter text & store spec lengths
    """
    # Store spectrogram lengths for Bucketing
    # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
    # spec_length = wav_length // hop_length

    lengths = []
    audiopath_and_text_new = []
    # 取出每一行的音频地址audiopath和音频内容text
    for spk, id_, phonemes, durations, pitch, energy in self.audiopaths_and_text:
      phn_dur = self.get_duration_flag(durations)
      if sum(phn_dur) > 1400:
        print("skip too long wav", spk, id_)
        continue
      # get_size获取文件大小（字节数），这里计算wav的长度，根据上方计算公式得出结果
      wav_path = f"dataset/{spk}/{id_}.wav"
      lengths.append(os.path.getsize(wav_path) // (2 * self.hop_length))
      sid = self.spk_map[spk]
      audiopath_and_text_new.append([wav_path, spk, id_, sid, phonemes, durations, pitch, energy])
    self.lengths = lengths
    self.audiopaths_and_text = audiopath_and_text_new

  def get_audio_text_pair(self, audiopath_and_text):
    # separate filename and text

    wav_path, spk, id_, sid, phonemes, durations, pitch, energy = audiopath_and_text

    phonemes = self.get_phonemes(phonemes)
    phn_dur = self.get_duration_flag(durations)
    spk = torch.LongTensor([int(sid)])
    # 得到文本内容、频谱图、音频数据
    spec, wav = self.get_audio(wav_path)
    f0 = torch.FloatTensor([float(i) for i in pitch.strip().split(" ")])
    energy = torch.FloatTensor([float(i) for i in energy.strip().split(" ")])
    sumdur = sum(phn_dur)
    assert abs(spec.shape[-1] - sumdur) < 2, wav_path
    if spec.shape[-1] > sumdur:
      spec = spec[:, :sumdur]
      wav = wav[:, :sumdur * self.hop_length]
    elif spec.shape[-1] < sumdur:
      spec_pad = torch.zeros([spec.shape[0], sumdur])
      wav_pad = torch.zeros([1, sumdur * self.hop_length])
      spec_pad[:, :spec.shape[-1]] = spec
      wav_pad[:, :wav.shape[-1]] = wav
      spec = spec_pad
      wav = wav_pad
    assert phonemes.shape == f0.shape == phn_dur.shape == energy.shape, wav_path
    assert sumdur == wav.shape[-1] // self.hop_length
    return phonemes, f0, phn_dur, spec, wav, spk, energy

  def get_phonemes(self, phonemes):
    phonemes_norm = cleaned_text_to_sequence(phonemes.split(" "))
    phonemes_norm = torch.LongTensor(phonemes_norm)
    return phonemes_norm

  def get_duration_flag(self, phn_dur):
    phn_dur = [int(i) for i in phn_dur.split(" ")]
    phn_dur = torch.LongTensor(phn_dur)
    return phn_dur

  def get_audio(self, filename):
    # 使用scipy.io.wavfile.read读取的音频文件
    audio, sampling_rate = load_wav_to_torch(filename)
    if sampling_rate != self.sampling_rate:
      raise ValueError("{} {} SR doesn't match target {} SR".format(
        sampling_rate, self.sampling_rate))
    audio_norm = audio / self.max_wav_value
    audio_norm = audio_norm.unsqueeze(0)
    spec_filename = filename.replace(".wav", ".spec.pt")
    if os.path.exists(spec_filename):
      fsize = os.path.getsize(spec_filename)
      if fsize == 0:
        print("spec_filesize: ", str(fsize) + "  |  " + spec_filename)
      spec = torch.load(spec_filename)
    else:
      spec = spectrogram_torch(audio_norm, self.filter_length,
                               self.sampling_rate, self.hop_length, self.win_length,
                               center=False)
      spec = torch.squeeze(spec, 0)
      torch.save(spec, spec_filename)
    return spec, audio_norm

  def __getitem__(self, index):
    return self.get_audio_text_pair(self.audiopaths_and_text[index])

  def __len__(self):
    return len(self.audiopaths_and_text)


class TextAudioSpeakerCollate():
  """ Zero-pads model inputs and targets
  """

  def __init__(self, return_ids=False):
    self.return_ids = return_ids

  def __call__(self, batch):
    """Collate's training batch from normalized text and aduio
    PARAMS
    ------
    batch: [text_normalized, spec_normalized, wav_normalized]
    """
    # Right zero-pad all one-hot text sequences to max input length
    _, ids_sorted_decreasing = torch.sort(
      torch.LongTensor([x[4].size(1) for x in batch]),
      dim=0, descending=True)
    # spec和wav都是tensor，所以可以用size取大小。其余字符串得用len函数取大小

    # phonemes,f0, phn_dur, spec, wav
    max_phonemes_len = max([len(x[0]) for x in batch])
    max_f0_len = max([len(x[1]) for x in batch])
    max_phndur_len = max([len(x[2]) for x in batch])
    max_spec_len = max([x[3].size(1) for x in batch])
    max_wav_len = max([x[4].size(1) for x in batch])
    max_energy_len = max([len(x[6]) for x in batch])

    phonemes_lengths = torch.LongTensor(len(batch))
    spec_lengths = torch.LongTensor(len(batch))
    wav_lengths = torch.LongTensor(len(batch))
    sid = torch.LongTensor(len(batch))

    phonemes_padded = torch.LongTensor(len(batch), max_phonemes_len)
    f0_padded = torch.FloatTensor(len(batch), max_f0_len)
    phndur_padded = torch.LongTensor(len(batch), max_phndur_len)
    spec_padded = torch.FloatTensor(len(batch), batch[0][3].size(0), max_spec_len)
    wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
    energy_padded = torch.FloatTensor(len(batch), max_energy_len)

    phonemes_padded.zero_()
    phndur_padded.zero_()
    spec_padded.zero_()
    wav_padded.zero_()
    f0_padded.zero_()
    energy_padded.zero_()

    for i in range(len(ids_sorted_decreasing)):
      row = batch[ids_sorted_decreasing[i]]

      phonemes = row[0]

      phonemes_padded[i, :phonemes.size(0)] = phonemes
      phonemes_lengths[i] = phonemes.size(0)

      f0 = row[1]

      f0_padded[i, :f0.size(0)] = f0

      phndur = row[2]
      phndur_padded[i, :phndur.size(0)] = phndur

      spec = row[3]
      spec_padded[i, :, :spec.size(1)] = spec
      spec_lengths[i] = spec.size(1)

      wav = row[4]
      wav_padded[i, :, :wav.size(1)] = wav
      wav_lengths[i] = wav.size(1)

      sid[i] = row[5]

      energy = row[6]

      energy_padded[i, :energy.size(0)] = energy

    # (phonemes, phonemes_lengths,
    #  f0,
    #  phndur,
    #  spec, spec_lengths, wav, wav_lengths)
    # print(f0_padded.shape, sum(phndur_padded[0,:]), phndur_padded[0,:])
    return phonemes_padded, phonemes_lengths, f0_padded, energy_padded, phndur_padded, \
      spec_padded, spec_lengths, \
      wav_padded, wav_lengths, sid


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
  """
  Maintain similar input lengths in a batch.
  Length groups are specified by boundaries.
  Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

  It removes samples which are not included in the boundaries.
  Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
  """

  def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
    super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
    self.lengths = dataset.lengths
    self.batch_size = batch_size
    self.boundaries = boundaries

    self.buckets, self.num_samples_per_bucket = self._create_buckets()
    self.total_size = sum(self.num_samples_per_bucket)
    self.num_samples = self.total_size // self.num_replicas

  def _create_buckets(self):
    # buckets = [[],[],[],[],[],...]
    buckets = [[] for _ in range(len(self.boundaries) - 1)]

    for i in range(len(self.lengths)):
      length = self.lengths[i]

      idx_bucket = self._bisect(length)
      if idx_bucket != -1:
        buckets[idx_bucket].append(i)

    for i in range(len(buckets) - 1, 0, -1):
      if len(buckets[i]) == 0:
        buckets.pop(i)
        self.boundaries.pop(i + 1)

    num_samples_per_bucket = []
    for i in range(len(buckets)):
      len_bucket = len(buckets[i])
      total_batch_size = self.num_replicas * self.batch_size
      rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
      num_samples_per_bucket.append(len_bucket + rem)
    return buckets, num_samples_per_bucket

  def __iter__(self):
    # deterministically shuffle based on epoch
    g = torch.Generator()
    g.manual_seed(self.epoch)

    indices = []
    if self.shuffle:
      for bucket in self.buckets:
        indices.append(torch.randperm(len(bucket), generator=g).tolist())
    else:
      for bucket in self.buckets:
        indices.append(list(range(len(bucket))))

    batches = []
    for i in range(len(self.buckets)):
      bucket = self.buckets[i]
      len_bucket = len(bucket)
      ids_bucket = indices[i]
      num_samples_bucket = self.num_samples_per_bucket[i]

      # add extra samples to make it evenly divisible
      rem = num_samples_bucket - len_bucket
      ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]

      # subsample
      ids_bucket = ids_bucket[self.rank::self.num_replicas]

      # batching
      for j in range(len(ids_bucket) // self.batch_size):
        batch = [bucket[idx] for idx in ids_bucket[j * self.batch_size:(j + 1) * self.batch_size]]
        batches.append(batch)

    if self.shuffle:
      batch_ids = torch.randperm(len(batches), generator=g).tolist()
      batches = [batches[i] for i in batch_ids]
    self.batches = batches

    assert len(self.batches) * self.batch_size == self.num_samples
    return iter(self.batches)

  def _bisect(self, x, lo=0, hi=None):
    if hi is None:
      hi = len(self.boundaries) - 1

    if hi > lo:
      mid = (hi + lo) // 2
      if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
        return mid
      elif x <= self.boundaries[mid]:
        return self._bisect(x, lo, mid)
      else:
        return self._bisect(x, mid + 1, hi)
    else:
      return -1

  def __len__(self):
    return self.num_samples // self.batch_size

#这是一个PyTorch数据集类，用于加载音频和文本文件，并将它们转化为整数序列和音频的语谱图（spectrogram）。具体来说：
#在__init__中，该类加载音频文件和对应的文本，其中hparams是一个包含一些配置参数的对象，
#如最大音频值（max_wav_value）、采样率（sampling_rate）、滤波器长度（filter_length）、跳跃长度（hop_length）
#和窗口长度（win_length）。此外，这个类还将音频文件名和文本随机排序，并调用_filter()方法对数据进行处理。

#filter()方法对文本数据进行处理，包括文本过滤、筛选和存储语谱图长度，以便进行bucketing（桶排序）。
#在这个代码中，桶排序用于在 DataLoader 中对音频文件和对应的文本进行排序和分批处理。对于 Speech-to-Text 任务中的文本和音频数据，
#它们的长度通常是不同的，因此我们需要将它们划分到不同的批次中，并根据它们的长度排序以提高训练效率。而桶排序是一种快速有效的排序方法，
#它将元素按照其值的范围划分为不同的桶，然后将桶中的元素进行排序，最终将桶中的元素合并为有序序列。在这个代码中，每个批次的长度在一定范围内，
#因此使用桶排序可以快速地将相似长度的音频和文本数据放到一起进行训练。

#get_audio_text_pair()方法对音频和文本数据进行处理，并返回文本内容、频谱图、音频数据和标签（标签是指说话人的编号）。

#get_phonemes()方法将原始文本转换为整数序列。
#将文本转换为整数序列是因为计算机只能处理数字，而不能直接处理文本。在自然语言处理领域中，
#我们通常会将文本转换为数字表示，这样可以方便地对文本进行处理和分析，比如分类、聚类、文本生成等任务。

#get_duration_flag()方法将原始持续时间转换为整数序列。

#get_audio()方法使用librosa库将音频文件读入到PyTorch张量中，并计算其语谱图。在计算语谱图之前，
#将音频数据进行归一化处理。如果之前已经计算过该音频文件的语谱图，那么就直接加载之前计算好的语谱图文件，
#否则就计算该音频文件的语谱图并保存下来以便下次使用。
#在音频信号处理中，归一化是一项非常重要的预处理步骤。音频信号通常由一系列的振幅值组成，
#这些值的范围可以是正数、负数或零。归一化的主要目的是将这些振幅值调整到一个标准范围内，以便于处理、分析和比较。
#归一化可以使不同的音频数据具有相同的振幅范围。这有助于消除振幅变化对数据处理和分析的影响，以及消除不同数据之间的比较难度。
#另外，归一化还可以避免在数字信号处理过程中出现数据溢出的问题。

#在具体的实践中，通常将音频数据的振幅值缩放到[-1,1]或[0,1]范围内。这可以通过除以最大振幅值或使用一些特定的归一化方法来实现。



