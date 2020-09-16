import torch

from torchsupport.training.gan import RothGANTraining

class MsiTraining(RothGANTraining):
  def each_generate(self, inputs, generated, sample):
    data, onehot = generated
    view = data[:30].detach().to("cpu")
    labels = onehot[0][:30].argmax(dim=1)

    for label in range(2):
      hit = (labels == label).view(-1).nonzero().view(-1)
      if hit.size(0):
        images = torch.cat([
          image
          for image in view[hit]
        ], dim=-1)
        kind = "mss" if label else "msi"
        self.writer.add_image(f"sample {kind}", images, self.step_id)

  def mixing_key(self, data):
    return data[0]
