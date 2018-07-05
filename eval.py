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


sentences1 = [
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

testSentence2 = {'hf15001':'J ian4 R en2 J iu4 Sh ix4 J iao3 Q ing2',
'hf15013':'F ang4 B u4 X ia4 R ong2 H ua2 F u4 G ui4 D e5 R en2 , J iu4 Y ong3 Y van3 Ch eng3 B u4 L iao3 D a4 Q i4 H ou4',
'hf15037':'Y ou3 Sh ix2 H ou4 B u4 Zh eng1 , B i3 N eng2 Zh eng1 H ui4 Zh eng1 Zh ix1 R en2 Y ou3 F u2 D uo1 L e5',
'hf15061':'M ei2 Y ou3 H uang2 Sh ang4 D e5 Zh ix3 Y i4 W o3 J ve2 B u4 J iu4 S iy3',
'hf15067':'B ie2 Sh uo1 B en3 G ong1 Y e3 H uai2 G uo4 L ong2 T ai1 , J iu4 S uan4 M ei2 H uai2 G uo4 , B en3 G ong1 Y e3 J ian4 D uo1 L e5 . D ao3 Sh ix4 M ei2 J ian4 G uo4 N i3 Zh e4 Y ang4 , H uai2 L e5 G e4 H ai2 Z iy5 X iang4 J ian3 L e5 G e4 Y van2 B ao3 Sh ix4 D e5 D ao4 Ch u4 X ian3 B ai3 , D ao4 D i3 Sh ix4 M ei2 J ian4 G uo4 Sh ix4 M ian4 D e5 X iao3 J ia1 Z iy3',
'hf15211':'H uang2 Sh ang4 D e5 Y i4 S iy1 Sh ix4 , J iao1 Y ou2 H uang2 H ou4 N iang2 N iang2 Q van2 Q van2 Z uo4 Zh u3',
'hf15217':'Zh en1 Sh ix4 D ui4 B u4 Zh u4 L e5 , Ch eng1 H u1 G uan4 L e5 N in2 N iang2 N iang2 , Zh ou4 R an2 Y ao4 J iao4 N i3 D a2 Y ing1 , H ai2 Zh en1 Sh ix4 B u4 X i2 G uan4 N e5',
'hf15223':'N i3 H u2 Sh uo1 , H uang2 Sh ang4 Z en3 M e5 H ui4 D ui4 W o3 M ei2 Y ou3 Zh en1 X in1',
'hs_zh_jj_lbx_03355':'W o3 D a3 S iy3 Y i1 G e4 W o1 K ou4 W o3 G ou4 B en3 , W o3 D a3 B u4 S iy3 W o1 K ou4 F an3 B ei4 W o1 K ou4 D a3 S iy3 W o3 X in1 G an1',
'hs_zh_jj_lbx_03356':'J ian1 L i4 D e5 H ei1 L iang4 Y a1 M ing2 C ong2 N a4 J v4 S iy3 Sh ix1 Sh ang4 X iang4 G ou1 W ai4 X iang3 Zh e5 , AA an4 H ong2 D e5 X ie3 W ei4 Ch ao2 Zh e5 G ou1 K ou3 G ou1 W ai4 X iang3 X iang3 L iang4 L iang4 D e5 M an4 Y i4',
}


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
  i=0
  for (k,text) in  testSentence2.items(): 
  #for i, text in enumerate(sentences):
    path = '%s-%d.wav' % (base_path, i)
    print('Synthesizing: %s' % path)
    i+=1
    with open(path, 'wb') as f:
      f.write(synth.synthesize(text,k))


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
