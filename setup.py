from setuptools import setup, find_packages

setup(name='deep_rl',
      packages=[package for package in find_packages()
                if package.startswith('deep_rl')],
      install_requires=[],
      python_requires='>=3.9',
      description="Modularized Implementation of Deep RL Algorithms",
      author="Shangtong Zhang",
      url='https://github.com/ShangtongZhang/DeepRL',
      author_email="zhangshangtong.cpp@gmail.com",
      version="1.6")
