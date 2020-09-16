import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

setuptools.setup(
  name="histogan",
  version="0.0.1",
  author="Michael Jendrusch",
  author_email="jendrusch@stud.uni-heidelberg.de",
  description="HistoGAN port to PyTorch.",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/mjendrusch/pytorch-histogan/",
  packages=setuptools.find_packages(),
  classifiers=(
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ),
  install_requires=[
    'torch',
    'torchvision',
    'numpy',
    'torchsupport @ git+https://github.com/mjendrusch/torchsupport@master',
  ]
)
