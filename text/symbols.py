pu_symbols = ['!', '?', '…', ",", ".", "sp"]
ja_symbols = [
  # japanese-common
  'ts.', 'f.', 'sh.', 'ry.', 'py.', 'h.', 'p.', 'N.', 'a.', 'm.', 'w.', 'ky.',
  'n.', 'd.', 'j.', 'cl.', 'ny.', 'z.', 'o.', 'y.', 't.', 'u.', 'r.', 'pau',
  'ch.', 'e.', 'b.', 'k.', 'g.', 's.', 'i.',
  # japanese-unique
  'gy.', 'my.', 'hy.', 'br', 'by.', 'v.', 'ty.', 'xx.', 'U.', 'I.', 'dy.'
]
en_symbols = [
  'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0',
  'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH',
  'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2',
  'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH'
]
zh_symbols = [
  'a1', 'a2', 'a3', 'a4', 'a5', 'ai1', 'ai2', 'ai3', 'ai4', 'ai5', 'air1', 'air2', 'air3', 'air4', 'air5', 'an1',
  'an2', 'an3', 'an4', 'an5', 'ang1', 'ang2', 'ang3', 'ang4', 'ang5', 'angr1', 'angr2', 'angr3', 'angr4', 'angr5',
  'anr1', 'anr2', 'anr3', 'anr4', 'anr5', 'ao1', 'ao2', 'ao3', 'ao4', 'ao5', 'aor1', 'aor2', 'aor3', 'aor4', 'aor5',
  'ar1', 'ar2', 'ar3', 'ar4', 'ar5', 'b', 'c', 'ch', 'd', 'e1', 'e2', 'e3', 'e4', 'e5', 'ei1', 'ei2', 'ei3', 'ei4',
  'ei5', 'eir1', 'eir2', 'eir3', 'eir4', 'eir5', 'en1', 'en2', 'en3', 'en4', 'en5', 'eng1', 'eng2', 'eng3', 'eng4',
  'eng5', 'engr1', 'engr2', 'engr3', 'engr4', 'engr5', 'enr1', 'enr2', 'enr3', 'enr4', 'enr5', 'er1', 'er2', 'er3',
  'er4', 'er5', 'f', 'g', 'h', 'i1', 'i2', 'i3', 'i4', 'i5', 'ia1', 'ia2', 'ia3', 'ia4', 'ia5', 'ian1', 'ian2', 'ian3',
  'ian4', 'ian5', 'iang1', 'iang2', 'iang3', 'iang4', 'iang5', 'iangr1', 'iangr2', 'iangr3', 'iangr4', 'iangr5',
  'ianr1', 'ianr2', 'ianr3', 'ianr4', 'ianr5', 'iao1', 'iao2', 'iao3', 'iao4', 'iao5', 'iaor1', 'iaor2', 'iaor3',
  'iaor4', 'iaor5', 'iar1', 'iar2', 'iar3', 'iar4', 'iar5', 'ie1', 'ie2', 'ie3', 'ie4', 'ie5', 'ier1', 'ier2', 'ier3',
  'ier4', 'ier5', 'ii1', 'ii2', 'ii3', 'ii4', 'ii5', 'iii1', 'iii2', 'iii3', 'iii4', 'iii5', 'iiir1', 'iiir2', 'iiir3',
  'iiir4', 'iiir5', 'iir1', 'iir2', 'iir3', 'iir4', 'iir5', 'in1', 'in2', 'in3', 'in4', 'in5', 'ing1', 'ing2', 'ing3',
  'ing4', 'ing5', 'ingr1', 'ingr2', 'ingr3', 'ingr4', 'ingr5', 'inr1', 'inr2', 'inr3', 'inr4', 'inr5', 'io1', 'io2',
  'io3', 'io4', 'io5', 'iong1', 'iong2', 'iong3', 'iong4', 'iong5', 'iongr1', 'iongr2', 'iongr3', 'iongr4', 'iongr5',
  'ior1', 'ior2', 'ior3', 'ior4', 'ior5', 'iou1', 'iou2', 'iou3', 'iou4', 'iou5', 'iour1', 'iour2', 'iour3', 'iour4',
  'iour5', 'ir1', 'ir2', 'ir3', 'ir4', 'ir5', 'j', 'k', 'l', 'm', 'n', 'o1', 'o2', 'o3', 'o4', 'o5', 'ong1', 'ong2',
  'ong3', 'ong4', 'ong5', 'ongr1', 'ongr2', 'ongr3', 'ongr4', 'ongr5', 'or1', 'or2', 'or3', 'or4', 'or5', 'ou1', 'ou2',
  'ou3', 'ou4', 'ou5', 'our1', 'our2', 'our3', 'our4', 'our5', 'p', 'q', 'r', 's', 'sh', 't', 'u1', 'u2', 'u3', 'u4',
  'u5', 'ua1', 'ua2', 'ua3', 'ua4', 'ua5', 'uai1', 'uai2', 'uai3', 'uai4', 'uai5', 'uair1', 'uair2', 'uair3', 'uair4',
  'uair5', 'uan1', 'uan2', 'uan3', 'uan4', 'uan5', 'uang1', 'uang2', 'uang3', 'uang4', 'uang5', 'uangr1', 'uangr2',
  'uangr3', 'uangr4', 'uangr5', 'uanr1', 'uanr2', 'uanr3', 'uanr4', 'uanr5', 'uar1', 'uar2', 'uar3', 'uar4', 'uar5',
  'uei1', 'uei2', 'uei3', 'uei4', 'uei5', 'ueir1', 'ueir2', 'ueir3', 'ueir4', 'ueir5', 'uen1', 'uen2', 'uen3', 'uen4',
  'uen5', 'ueng1', 'ueng2', 'ueng3', 'ueng4', 'ueng5', 'uengr1', 'uengr2', 'uengr3', 'uengr4', 'uengr5', 'uenr1',
  'uenr2', 'uenr3', 'uenr4', 'uenr5', 'uo1', 'uo2', 'uo3', 'uo4', 'uo5', 'uor1', 'uor2', 'uor3', 'uor4', 'uor5', 'ur1',
  'ur2', 'ur3', 'ur4', 'ur5', 'v1', 'v2', 'v3', 'v4', 'v5', 'van1', 'van2', 'van3', 'van4', 'van5', 'vanr1', 'vanr2',
  'vanr3', 'vanr4', 'vanr5', 've1', 've2', 've3', 've4', 've5', 'ver1', 'ver2', 'ver3', 'ver4', 'ver5', 'vn1', 'vn2',
  'vn3', 'vn4', 'vn5', 'vnr1', 'vnr2', 'vnr3', 'vnr4', 'vnr5', 'vr1', 'vr2', 'vr3', 'vr4', 'vr5', 'x', 'z', 'zh'
]
symbols = ["_"] + zh_symbols + ja_symbols + en_symbols + pu_symbols
