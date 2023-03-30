import torch


def feature_loss(fmap_r, fmap_g):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = torch.mean((1 - dr) ** 2)
    g_loss = torch.mean(dg ** 2)
    loss += (r_loss + g_loss)
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


def generator_loss(disc_outputs):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1 - dg) ** 2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()

  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l
#这段代码是一个基于PyTorch的深度学习模型的损失函数的实现，其中包括以下四个不同的损失函数：
#1.feature_loss(fmap_r, fmap_g)：计算两个特征映射（feature map）之间的L1距离损失（L1 distance loss）。
#2.discriminator_loss(disc_real_outputs, disc_generated_outputs)：计算判别器（discriminator）的损失函数，包括真实数据的损失和生成数据的损失，使用的是均方误差损失（mean squared error loss）。
#3.generator_loss(disc_outputs)：计算生成器（generator）的损失函数，同样使用均方误差损失。
#4.kl_loss(z_p, logs_q, m_p, logs_p, z_mask)：计算KL散度（Kullback-Leibler divergence）损失，用于对抗性生成模型（adversarial generative models）的训练中，以保证生成器的输出与给定分布的相似性。