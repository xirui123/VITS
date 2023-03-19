# ViSpeech

## 相比于原版 VITS

+ 将 FastSpeech2 的 VarianceAdapter 结构添加进了 VITS
+ 删除了 Monotonic Alignment，使用 MFA 对齐后输入时长
+ 添加了音素级 F0Predictor，EnergyPredictor
+ 可以逐音素手动编辑音高、音量和时长
+ 添加了 FramePriorNetwork
+ 使用 PaddleSpeech 作为中文文本前端，实现更可靠的文本正则化以及 G2P
+ 采样率使用 44100 HZ
