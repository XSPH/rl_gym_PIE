from setuptools import find_packages
from distutils.core import setup

setup(name='unitree_rl_gym',
      version='1.0.0',
      author='Unitree Robotics',
      license="BSD-3-Clause",
      packages=find_packages(include=['legged_gym', 'legged_gym.*']),
      author_email='support@unitree.com',
      description='Template RL environments for Unitree Robots',
      install_requires=['isaacgym', 'rsl_rl==1.0.2', 'warp-lang==1.6.2', 'matplotlib', 'numpy>=1.21,<1.24', 'tensorboard', 'mujoco==3.2.3', 'pyyaml'])
