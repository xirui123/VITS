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

#��δ�����Ҫ�Ƕ�����ʶ�����ݽ��д�������ͬ���Ե�ѵ�������б�ϲ���������˵���˽��仮��Ϊѵ��������֤����
#���������� defaultdict ���ݽṹ�����ڴ���һ���ֵ䣬�����ʲ����ڵļ�ֵ��ʱ�����Զ�����һ��Ĭ��ֵ����������Ĭ��ֵΪһ�����б�
#������һ�� langs �б�������Ҫ������������͡�
#����һ����Ϊ spk2utts ���ֵ䣬���ڴ洢ÿ��˵���ˣ�speaker����˵�����о��ӣ�utterance�������ʼ��ֵΪһ�����б�
#����ÿ���������ͣ���ȡ���Ӧ��ѵ���б��ļ���������ÿһ�У���ȡ��ÿ������������˵������Ϣ spk���������о�����ӵ� spk2utts �ж�Ӧ˵���˵��б��С�
#��������������һ�� val_n_per_spk ��������������ÿ��˵���˵Ĳ��־�����������Щ���ӽ���Ϊ��֤����
#���ţ��������������б� val_lines �� train_lines�����ڴ洢���յ���֤����ѵ���������б�
#���� spk2utts ��ÿ��˵���˼������о��ӣ����䰴�� val_n_per_spk ����������Ϊ��֤����ѵ�������������ֺ���б�ֱ���ӵ� val_lines �� train_lines �С�
#��󣬽�ѵ��������֤�����б�ֱ�д�뵽 train.list �� val.list �ļ��С�