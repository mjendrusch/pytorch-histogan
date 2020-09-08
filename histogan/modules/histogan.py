r"""This module implements a parametric histoGAN / DCGAN architecture suitable for generating histology images."""

import torch
import torch.nn as nn
import torch.nn.functional as func

def histo_generator_layer(position, in_channels, out_channels, kernel_size, activation):
  return nn.Sequential(
    nn.ConvTranspose2d(
      in_channels,
      out_channels,
      kernel_size,
      padding=kernel_size // 2 if position else 0,
      output_padding=1 if position else 0,
      stride=2
    ),
    nn.BatchNorm2d(out_channels),
    activation
  )

class HistoGenerator(nn.Module):
  r"""Configurable HistoGAN generator, based on the Matlab HistoGAN implementation.

  Args:
    base_channels (int): base number of channels for each layer of the generator. This
      will be muliplied by ``channel_factors`` at each layer.
    channel_factors (List[int]): factors by which to multiply the number of filters in
      each layer of the generator.
    kernel_size (int): size of the transpose convolution kernel at each layer of the generator.
    condition_size (int): number of classes for conditional generation.
    latent_size (int): size of the normal-distributed latent variable.
    condition_embedding_size (int): size of the condition embedding.
    embed_first (bool): mode of including the condition embedding into the generator.
      If ``True``, embeds the condition and concatenates with the latent vector, followed by
      a linear layer returning the initial 4x4 feature map. If ``False``, generates a 4x4 feature
      map from the latent vector and a separate 4x4 feature map from the condition and concatenates
      both of these feature maps.
    activation (callable): a nonlinear activation function. Defaults to ReLU.
    target_size (int): desired size of the output image.
  """
  def __init__(self, base_channels=64, channel_factors=None, kernel_size=5,
               condition_size=2, latent_size=100, condition_embedding_size=100,
               embed_first=True, activation=None, target_size=224):
    super().__init__()
    self.embed_first = embed_first
    self.embed_condition = ...
    self.target_size = target_size
    self.latent_size = latent_size
    self.condition_size = condition_size

    total_embedding_size = ...
    if embed_first:
      total_embedding_size = latent_size + condition_embedding_size
      self.embed_condition = nn.Linear(
        condition_size,
        condition_embedding_size
      )
    else:
      total_embedding_size = latent_size
      self.embed_condition = nn.Linear(
        condition_size,
        condition_embedding_size * 4 ** 2
      )

    self.activation = activation or nn.ReLU()
    self.initial = nn.Linear(
      total_embedding_size,
      base_channels * channel_factors[0] * 4 ** 2
    )
    self.blocks = nn.ModuleList([
      histo_generator_layer(
        position=idx,
        in_channels=in_factor * base_channels,
        out_channels=out_factor * base_channels,
        kernel_size=kernel_size,
        activation=self.activation
      )
      for idx, (in_factor, out_factor) in enumerate(zip(
        channel_factors[:-1], channel_factors[1:]
      ))
    ])
    self.postprocess = nn.ConvTranspose2d(
      base_channels * channel_factors[-1], 3, kernel_size
    )

  def sample(self, batch_size):
    noise = torch.randn(batch_size, self.latent_size)
    condition = torch.randint(0, self.condition_size, (batch_size,))
    one_hot = torch.zeros(batch_size, self.condition_size)
    one_hot[torch.arange(batch_size), condition] = 1
    return noise, one_hot

  def forward(self, data):
    latent, condition = data
    out = ...
    if self.embed_first:
      out = torch.cat((
        latent,
        self.embed_condition(condition)
      ), dim=1)
      out = self.initial(out).view(out.size(0), -1, 4, 4)
    else:
      out = self.initial(latent).view(out.size(0), -1, 4, 4)
      cond = self.embed_condition(condition).view(out.size(0), -1, 4, 4)
      out = torch.cat((out, cond), dim=1)

    for block in self.blocks:
      out = block(out)
    out = self.postprocess(out)
    start = (out.size(-1) - self.target_size) // 2
    stop = start + self.target_size
    out = out[:, :, start:stop, start:stop]

    return (out.tanh() + 1) / 2, condition

def histo_discriminator_layer(in_channels, out_channels, kernel_size, activation):
  return nn.Sequential(
    nn.Conv2d(
      in_channels, out_channels, kernel_size,
      padding=kernel_size // 2, stride=2
    ),
    nn.BatchNorm2d(out_channels),
    activation
  )

class HistoDiscriminator(nn.Module):
  r"""Configurable HistoGAN discriminator, based on the Matlab HistoGAN implementation.

  Args:
    base_channels (int): base number of channels for each layer of the discriminator. This
      will be muliplied by ``channel_factors`` at each layer.
    channel_factors (List[int]): factors by which to multiply the number of filters in
      each layer of the discriminator.
    kernel_size (int): size of the transpose convolution kernel at each layer of the discriminator.
    condition_size (int): number of classes for conditional generation.
    condition_embedding_size (int): size of the condition embedding.
    activation (callable): a nonlinear activation function. Defaults to ReLU.
    drop (range(0, 1)): dropout rate in the first layer of the discriminator.
    mode ("first", "last", "classifier"): mode of operation of the discriminator.
      "first" broadcasts and concatenates the condition embedding directly to the input.
      "last" concatenates the condition embedding to the last feature map of the discriminator.
      "classifier" produces one discriminator output for each class used for conditional
      generation and returns the value at the input condition.
  """
  def __init__(self, base_channels=64, channel_factors=None, kernel_size=5,
               condition_size=2, condition_embedding_size=100, activation=None,
               drop=0.5, mode="last"):
    super().__init__()
    self.mode = mode
    self.drop = nn.Dropout(drop)
    self.activation = activation or nn.LeakyReLU(0.2)
    self.preprocess = ...
    self.decision = ...

    self.condition_embedding = nn.Linear(condition_size, condition_embedding_size)

    if mode == "first":
      self.preprocess = histo_discriminator_layer(
        in_channels=3 + condition_embedding_size,
        out_channels=base_channels * channel_factors[0],
        kernel_size=kernel_size,
        activation=self.activation
      )
      self.decision = nn.Linear(base_channels * channel_factors[-1], 1)
    elif mode in ["last", "classifier"]:
      self.preprocess = histo_discriminator_layer(
        in_channels=3,
        out_channels=base_channels * channel_factors[0],
        kernel_size=kernel_size,
        activation=self.activation
      )
      if mode == "last":
        self.decision = nn.Linear(
          base_channels * channel_factors[-1] + condition_embedding_size, 1
        )
      else:
        self.decision = nn.Linear(
          base_channels * channel_factors[-1],
          condition_embedding_size
        )
    else:
      raise ValueError(
        f"Invalid mode '{mode}'. Possible modes are 'first', 'last' and 'classifier'."
      )

    self.blocks = nn.ModuleList([
      histo_discriminator_layer(
        in_channels=in_factor * base_channels,
        out_channels=out_factor * base_channels,
        kernel_size=kernel_size,
        activation=activation
      )
      for idx, (in_factor, out_factor) in enumerate(zip(
        channel_factors[:-1], channel_factors[1:]
      ))
    ])

  def forward(self, data):
    inputs, condition = data
    inputs = 2 * inputs - 1
    out = self.drop(inputs)
    if self.mode == "first":
      cond = self.embed_condition(condition)
      cond = cond[:, :, None, None].expand(*cond.shape, *inputs.shape[2:])
      out = torch.cat((out, cond), dim=1)
    out = self.preprocess(out)
    for block in self.blocks:
      out = block(out)
    out = func.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)

    if self.mode == "classifier":
      out = self.decision(out)
      ind = torch.arange(out.size(0), device=out.device)
      out = out[ind, condition.argmax(dim=1)]
    else:
      cond = self.embed_condition(condition)
      out = torch.cat((out, cond), dim=1)
      out = self.decision(out)

    return out

def histoGAN224(mode="last", embed_first=True, condition_size=2, condition_embedding_size=100):
  f"""Configuration used for 224x224 image generation.
  For details see :class:`HistoGenerator` and :class:`HistoDiscriminator`
  """
  gen = HistoGenerator(
    channel_factors=[8, 6, 4, 2, 1],
    embed_first=embed_first,
    condition_size=condition_size,
    condition_embedding_size=condition_embedding_size,
    target_size=224
  )
  disc = HistoDiscriminator(
    channel_factors=[1, 2, 4, 6, 8],
    condition_size=condition_size,
    condition_embedding_size=condition_embedding_size,
    mode=mode
  )
  return gen, disc
