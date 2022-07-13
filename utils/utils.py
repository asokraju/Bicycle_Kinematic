import numpy as np
from itertools import permutations
import pandas as pd
import os
import gym
from pandas.core.frame import DataFrame
# import seaborn as sns
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
# from sklearn.metrics import confusion_matrix


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).
    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

def train_model(steps):
    log_dir = "C:\\Users\\kkosara\\Bicycle_Kinematic\\results\\"
    tensorboard_log = "C:\\Users\\kkosara\\Bicycle_Kinematic\\results\\board"
    env_kwargs = {
        "discretization_steps":100, 
        "rand_initial_pos":True, 
        'obstacle_radius':[1., 1.]
    }
    gym.envs.register(
     id='BicycleKin-v0',
     entry_point='gym_bicycle.envs:BicycleKin'
    ) 
    env = gym.make('BicycleKin-v0', **env_kwargs)
    env = Monitor(env, log_dir)
    model = PPO(
        'MlpPolicy', 
        env, 
        verbose=0, 
        tensorboard_log=tensorboard_log, 
        seed = 1234)
    
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(steps, tb_log_name="test_1", callback=callback)
    return model