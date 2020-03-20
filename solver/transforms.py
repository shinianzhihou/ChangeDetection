import random

from torchvision import transforms
from torchvision.transforms import functional as tvF

class ToTensor(transforms.ToTensor):
  def __call__(self,pics):
    return [tvF.to_tensor(pic) for pic in pics]

class ToPILImage(transforms.ToPILImage):
  def __call__(self,imgs):
    return [tvF.to_pil_image(img,self.mode) for img in imgs]

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
  def __call__(self,imgs):
    if random.random() < self.p:
      return [tvF.hflip(img) for img in imgs]
    return imgs

class RandomVerticalFlip(transforms.RandomVerticalFlip):
  def __call__(self,imgs):
    if random.random() < self.p:
      return [tvF.vflip(img) for img in imgs]
    return imgs

class RandomRotation(transforms.RandomRotation):
  def __call__(self,imgs):
    angle = self.get_params(self.degrees)
    return [img.rotate(angle,self.resample,self.expand,self.center,fillcolor=self.fill) for img in imgs]

