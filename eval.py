import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


#sentences = [
  # From July 8, 2017 New York Times:
#  'Scientists at the CERN laboratory say they have discovered a new particle.',
#  'There’s a way to measure the acute emotional intelligence that has never gone out of style.',
#  'President Trump met with other leaders at the Group of 20 conference.',
#  'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
  # From Google's Tacotron example page:
#  'Generative adversarial network or variational auto-encoder.',
#  'The buses aren\'t the problem, they actually provide a solution.',
#  'Does the quick brown fox jump over the lazy dog?',
#  'Talib Kweli confirmed to AllHipHop that he will be releasing an album in the next year.',
#]

sentences0 = [
  'ta1 jing3 ti4 de5 xia4 le5 chuang2,',
  'gei3 liang3 ge5 sun1 zi5 ye4 hao3 bei4 zi5,',
  'you4 na2 guo4 yi1 ba3 da4 yi3 zi5, ba3 jie3 mei4 lia3 dang3 zhu4 gang1 zou3 dao4 ke4 ting1 jiu4 bei4 ren2 lan2 yao1 bao4 zhu4 le5.',
  'wei1 xin4 zhi1 fu4 zhang1 xiao3 long2 han3 jian4 lou4 mian4.',
  'cheng1 wei1 xin4 bu4 hui4 cha2 kan4 yong4 hu4 liao2 tian1 ji4 lu4 yi4 si an4 feng4 zhi1 fu4 bao3,',
  'ben3 wen2 lai2 zi4 teng2 xun4 ke1 ji4 .',
  'da4 hui4 zhi3 re4 nao5 tou2 liang3 tian1,',
  'yue4 hou4 yue4 song1 kua3 zui4 zhong1 chu1 ben3 lun4 wen2 ji2, jiu4 suan4 yuan2 man3 wan2 cheng2 ren4 wu5.',
  'lian2 dui4 zhi3 liu2 xia4 yi4 ming2 zhi2 ban1 yuan2, chui1 shi4 yuan2, si4 yang3 yuan2, wei4 sheng1 yuan2 deng3 ye3 lie4 dui4 pao3 bu4 gan2 wang3 zai1 qu1.',
  'yi1 jiu3 wu3 ling2 nian2 ba1 yue4 zhong1 yang1 ren2 min2 zheng4 fu3 zheng4 wu4 yuan4 ban1 bu4 le5,',
  'bao3 zhang4 fa1 ming2 quan2 yu3 zhuan1 li4 quan2 zan4 xing2 tiao2 li4.',
]

sentences = [
    'P ai4 D a4 X ing1 , N i3 Sh ix4 H ai3 M ian2 B ao3 B ao1 D e5 H ao3 P eng2 Y ou3 .',
    'J in1 T ian1 T ian1 Q i4 Zh en1 H ao3 AA a5 . N i3 J iu4 Sh ix4 G e4 B a4 G e3 .',
    'AA ai4 Q ing2 S an1 Sh ix2 L iu4 J i4 , AA ai1 Y o5 AA ai1 Y o5 AA ai1 Y o5 .',
    'EE er4 L e5 B a1 J i1 D e5 W o3 , H e2 EE er4 L e5 B a1 J i1 D e5 N i3 , Zh en1 Sh ix4 J ve2 P ei4 AA a5 .',
    'T ian1 AA a5 , Zh e4 Sh ix4 Sh ang4 Z en3 M e5 H ui4 Y ou3 N i3 Zh e4 Zh ong3 Sh a3 B i1 . X in1 Q ing2 B u4 H ao3 , L an3 D e2 Sh uo1 N i3 .',
    'T ian1 K ong1 Y i1 Sh eng1 J v4 X iang3 , L ao3 Z iy3 Sh an3 L iang4 D eng1 Ch ang3 .',
    'K ai1 Sh en2 M e5 W an2 X iao4 , W o3 C ai2 B u4 Y ao4 G en1 N i3 P ai1 T uo1 N e5 .',
    'R u2 G uo3 AA a5 W o3 N eng2 AA a5 Z ai4 C iy3 K e4 AA a5 T u3 Ch u1 L ai2 AA a5 J iu4 B u4 H ui4 Y ou3 N a4 M e5 D uo1 AA ai4 B u4 AA ai4 D e5 Sh ix4 Q ing2 L e5 B a1 ,',
    'X in1 H ao3 L ei4 N e5 , AA ao2 W u1 AA ao2 AA ao2 W u2 .' ,
]

def get_output_base_path(checkpoint_path):
  base_dir = os.path.dirname(checkpoint_path)
  m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
  name = 'eval-%d' % int(m.group(1)) if m else 'eval'
  return os.path.join(base_dir, name)


def run_eval(args):
  print(hparams_debug_string())
  synth = Synthesizer()
  synth.load(args.checkpoint)
  base_path = get_output_base_path(args.checkpoint)
  for i, text in enumerate(sentences):
    path = '%s-%d.wav' % (base_path, i)
    print('Synthesizing: %s' % path)
    with open(path, 'wb') as f:
      f.write(synth.synthesize(text))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
  hparams.parse(args.hparams)
  run_eval(args)


if __name__ == '__main__':
  main()
