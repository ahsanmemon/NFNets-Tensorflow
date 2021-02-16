from setuptools import setup, find_packages

setup(
  name = 'nfnets-keras',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'NFNets, Keras',
  author = 'Yam Peleg',
  author_email = 'ypeleg2@gmail.com',
  url = 'https://github.com/ypeleg/nfnets-keras',
  keywords = [
    'computer vision',
    'image classification',
    'keras',
    'tensorflow',
    'adaptive gradient clipping'
  ],
  install_requires=[
    'keras',
    'tensorflow',
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)