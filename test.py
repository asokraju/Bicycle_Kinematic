import datetime
import os
from os import walk, listdir
import numpy as np
import gym
import time
from stable_baselines3 import PPO
# import pandas as pd
# import json
#local modules
from utils.utils import train_model

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':
    model = train_model(int(1e5))