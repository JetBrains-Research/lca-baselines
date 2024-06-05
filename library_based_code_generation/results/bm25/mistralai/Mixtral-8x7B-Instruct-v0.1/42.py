 ```python
import Acme
from Acme import datasets, networks, agents, core
from Acme.tf import savers, specs, types
import sonnet as snt
import tensorflow as tf
import functools
import numpy as np
import gym

def create_crr_agent(environment_spec: specs.EnvironmentSpec,
                     batch_size: int,
                     random_seed: int,
                     demonstrations_dataset: types.Dataset,
                     number_of_demonstration_episodes: int,
                