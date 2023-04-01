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

#��δ����������޸������ļ��ģ���Ҫ��Ϊ���¼������裺

#��ȡ�����ļ� configs/config.json �е����ݣ�ʹ�� json.load() ��������ת��Ϊ�ֵ���ʽ���洢�� config �����С�

#����һ�����ֵ� spk2id�������洢˵���˺Ͷ�Ӧ�� id��

#����ѵ�������ļ� filelists/train.list �е�ÿһ�У���ȡ��ǰ�е�˵���� spk�������˵���˻�δ���ֹ���������뵽 spk2id �ֵ��У����丳��һ���µ� id ֵ��

#�� spk2id �ֵ�洢�� config �����е� data �ֵ�� spk2id ���¡�

#���޸ĺ�� config ��������д�ص������ļ� configs/config.json �У�ʹ�� json.dump() �������ֵ�ת��Ϊ JSON ��ʽд���ļ��С����� indent=2 ������ʾ��� JSON ʱʹ�������ո�������