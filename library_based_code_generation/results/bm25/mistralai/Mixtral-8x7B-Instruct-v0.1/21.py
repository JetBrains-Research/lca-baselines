 ```python
import matplotlib.pyplot as plt
import seaborn as sns
from rlberry.agents.bandits import UCB, UCBV, ETC, MOSS, IMED, NPTS, EXP3
from rlberry.envs.bandits import BernoulliBandit
from rlberry.replay_buffers import ReplayBuffer
