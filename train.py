from math import log2, ceil

from histogan.data import GANData
from histogan.training import MsiTraining
from histogan.modules.histogan import get_histo_GAN
from histogan.parser import parse_args

from torchsupport.structured import DataParallel as SDP

if __name__ == "__main__":
  opt = parse_args()

  data = GANData(opt.path)
  size = data[0][0].size(-1) # get size from the dataset.

  generator, discriminator = get_histo_GAN(
    size,
    mode=opt.mode,
    condition_size=len(data.kinds),
    condition_embedding_size=opt.condition
  )

  if opt.device != "cpu":
    generator = SDP(generator)
    discriminator = SDP(discriminator)

  training = MsiTraining(
    generator, discriminator, data,
    smoothing=opt.smoothing,
    gamma=opt.gradient_penalty,
    generator_optimizer_kwargs=dict(
      lr=opt.generator_lr, betas=(opt.beta_1, opt.beta_2)
    ),
    discriminator_optimizer_kwargs=dict(
      lr=opt.discriminator_lr, betas=(opt.beta_1, opt.beta_2)
    ),
    max_epochs=opt.max_epochs,
    device=opt.device,
    batch_size=opt.batch_size,
    verbose=True,
    path_prefix=opt.prefix,
    network_name=opt.name
  )

  training.train()
