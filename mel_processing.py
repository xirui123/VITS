import os
import torch
import torch.utils.data
import numpy as np
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
  """
  PARAMS
  ------
  C: compression factor
  """
  return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
  """
  PARAMS
  ------
  C: compression factor used to compress
  """
  return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
  output = dynamic_range_compression_torch(magnitudes)
  return output


def spectral_de_normalize_torch(magnitudes):
  output = dynamic_range_decompression_torch(magnitudes)
  return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
  if torch.min(y) < -1.:
    print('min value is ', torch.min(y))
  if torch.max(y) > 1.:
    print('max value is ', torch.max(y))

  global hann_window
  dtype_device = str(y.dtype) + '_' + str(y.device)
  wnsize_dtype_device = str(win_size) + '_' + dtype_device
  if wnsize_dtype_device not in hann_window:
    hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

  y = torch.nn.functional.pad(y.unsqueeze(1), (
  int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode='reflect')
  y = y.squeeze(1)

  spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                    center=center, pad_mode='reflect', normalized=False, onesided=True)

  spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
  return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
  global mel_basis
  dtype_device = str(spec.dtype) + '_' + str(spec.device)
  fmax_dtype_device = str(fmax) + '_' + dtype_device
  if fmax_dtype_device not in mel_basis:
    mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
    mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
  spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
  spec = spectral_normalize_torch(spec)
  return spec


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
  if torch.min(y) < -1.:
    print('min value is ', torch.min(y))
  if torch.max(y) > 1.:
    print('max value is ', torch.max(y))

  global mel_basis, hann_window
  dtype_device = str(y.dtype) + '_' + str(y.device)
  fmax_dtype_device = str(fmax) + '_' + dtype_device
  wnsize_dtype_device = str(win_size) + '_' + dtype_device
  if fmax_dtype_device not in mel_basis:
    mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
    mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
  if wnsize_dtype_device not in hann_window:
    hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

  y = torch.nn.functional.pad(y.unsqueeze(1), (
  int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode='reflect')
  y = y.squeeze(1)

  spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                    center=center, pad_mode='reflect', normalized=False, onesided=True)

  spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

  spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
  spec = spectral_normalize_torch(spec)

  return spec
#dynamic_range_compression_torch(x, C=1, clip_val=1e-5): 对输入张量进行动态范围压缩。
#C参数是一个缩放因子，用于缩放输入张量的范围。clip_val参数是一个阈值，
#用于截取输入张量中小于该值的部分并将其设置为该值。

#dynamic_range_decompression_torch(x, C=1): 对输入张量进行动态范围解压缩。C参数是压缩时使用的缩放因子。

#spectral_normalize_torch(magnitudes): 对输入的频谱张量进行范围归一化。

#spectral_de_normalize_torch(magnitudes): 对输入的经过范围归一化的频谱张量进行反归一化。

#spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False): 将音频波形转换为频谱图。

#spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax): 将给定的频谱图转换为梅尔频率倒谱系数。

#mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False): 将音频波形转换为梅尔频谱图。