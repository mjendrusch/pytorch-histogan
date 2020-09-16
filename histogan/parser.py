from argparse import ArgumentParser

def parse_args():
  parser = ArgumentParser(
    description='Train a generative adversarial network on H&E-stained tumor tissue.'
  )
  parser.add_argument('path', metavar='PATH', type=str,
                      help='path to the training dataset.')
  parser.add_argument('--batch-size', type=int, nargs='?', default=32,
                      help='training batch size.')
  parser.add_argument('--condition', type=int, nargs='?', default=100,
                      help='size of the condition embedding.')
  parser.add_argument('--max-epochs', type=int, nargs='?', default=100,
                      help='training batch size.')
  parser.add_argument('--smoothing', type=float, nargs='?', default=0.1,
                      help='amount of label smoothing.')
  parser.add_argument('--gradient-penalty', type=float, nargs='?', default=1.0,
                      help='weight of the discriminator gradient penalty.')
  parser.add_argument('--generator-lr', type=float, nargs='?', default=2e-4,
                      help='generator learning rate.')
  parser.add_argument('--discriminator-lr', type=float, nargs='?', default=2e-4,
                      help='discriminator learning rate.')
  parser.add_argument('--beta-1', type=float, nargs='?', default=0.5,
                      help='Adam beta_1.')
  parser.add_argument('--beta-2', type=float, nargs='?', default=0.999,
                      help='Adam beta_2.')
  parser.add_argument('--device', type=str, nargs='?', default='cuda:0',
                      help='device to use for training (e.g. cpu, cuda:0).')
  parser.add_argument('--mode', type=str, nargs='?', default='last',
                      help='type of discriminator to train (fist, last, classifier).')
  parser.add_argument('--prefix', type=str, nargs='?', default='.',
                      help='path prefix for saving checkpoints and logs.')
  parser.add_argument('--name', type=str, nargs='?', default='histogan',
                      help='path ending for saving checkpoints and logs.')

  return parser.parse_args()
