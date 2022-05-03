from setuptools import setup

setup(name='gym_tendulum',
      version='0.0.1',
      install_requires=[
            'gym',
            'numpy',
            'scipy',
            'pygame',
            'pygame_recorder @ git+https://github.com/jcranney/pygame_recorder'
      ]
)