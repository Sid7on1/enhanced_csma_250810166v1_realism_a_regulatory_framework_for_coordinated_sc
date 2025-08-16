import os
import logging
from typing import Dict, List
import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config(ABC):
    """
    Base configuration class with common settings and methods.
    """

    def __init__(self, config_dict: Dict):
        self.config = config_dict
        self._validate_config()
        self._setup_device()
        self._setup_random_seed()

    @abstractmethod
    def _validate_config(self):
        """
        Validate the configuration dictionary. Raise ValueError for invalid configurations.
        """
        pass

    def _setup_device(self):
        """
        Set the device for torch based on the configuration.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.config['use_cuda'] else "cpu")
        logger.info(f"Device set to: {self.device}")

    def _setup_random_seed(self):
        """
        Set the random seed for reproducibility.
        """
        self.random_seed = self.config['random_seed']
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        logger.info(f"Random seed set to: {self.random_seed}")

    def save_config(self, file_path: str):
        """
        Save the configuration to a file.

        Args:
            file_path (str): Path to the file where the configuration will be saved.
        """
        with open(file_path, 'w') as file:
            file.write(self.config_to_str())
        logger.info(f"Configuration saved to: {file_path}")

    def config_to_str(self) -> str:
        """
        Convert the configuration dictionary to a string for saving or logging.

        Returns:
            str: String representation of the configuration.
        """
        return str(self.config)

class AgentConfig(Config):
    """
    Configuration class for the agent.
    """

    def __init__(self, config_dict: Dict):
        super().__init__(config_dict)
        self.env_name = self.config['env_name']
        self.state_size = self.config['state_size']
        self.action_size = self.config['action_size']
        self.learning_rate = self.config['learning_rate']
        self.buffer_size = self.config['buffer_size']
        self.batch_size = self.config['batch_size']
        self.hidden_sizes = self.config['hidden_sizes']
        self.num_hidden_layers = len(self.hidden_sizes) - 1
        self.tau = self.config['tau']
        self.gamma = self.config['gamma']
        self.update_every = self.config['update_every']
        self.num_updates = self.config['num_updates']
        self.target_update_interval = self.config['target_update_interval']
        self.noise_scale = self.config['noise_scale']
        self.noise_reduction = self.config['noise_reduction']
        self._setup_action_noise()

    def _validate_config(self):
        """
        Validate the configuration dictionary. Raise ValueError for invalid configurations.
        """
        if 'env_name' not in self.config:
            raise ValueError("Environment name not specified in the configuration.")
        if 'state_size' not in self.config:
            raise ValueError("State size not specified in the configuration.")
        if 'action_size' not in self.config:
            raise ValueError("Action size not specified in the configuration.")
        if 'learning_rate' not in self.config:
            raise ValueError("Learning rate not specified in the configuration.")
        if 'buffer_size' not in self.config:
            raise ValueError("Buffer size not specified in the configuration.")
        if 'batch_size' not in self.config:
            raise ValueError("Batch size not specified in the configuration.")
        if 'hidden_sizes' not in self.config:
            raise ValueError("Hidden layer sizes not specified in the configuration.")
        if 'tau' not in self.config:
            raise ValueError("Tau value for soft update not specified in the configuration.")
        if 'gamma' not in self.config:
            raise ValueError("Discount factor not specified in the configuration.")
        if 'update_every' not in self.config:
            raise ValueError("Update frequency not specified in the configuration.")
        if 'num_updates' not in self.config:
            raise ValueError("Number of updates not specified in the configuration.")
        if 'target_update_interval' not in self.config:
            raise ValueError("Target network update interval not specified in the configuration.")
        if 'noise_scale' not in self.config:
            raise ValueError("Action noise scale not specified in the configuration.")
        if 'noise_reduction' not in self.config:
            raise ValueError("Action noise reduction factor not specified in the configuration.")

    def _setup_action_noise(self):
        """
        Set up the action noise based on the specified scale and reduction factor.
        """
        self.action_noise = self.noise_scale * self.noise_reduction ** (
            self.update_every * self.num_updates / self.target_update_interval
        )
        logger.info(f"Action noise scale: {self.action_noise}")

class EnvironmentConfig(Config):
    """
    Configuration class for the environment.
    """

    def __init__(self, config_dict: Dict):
        super().__init__(config_dict)
        self.env_name = self.config['env_name']
        self.observation_space = self.config['observation_space']
        self.action_space = self.config['action_space']
        self.reward_range = self.config['reward_range']
        self.simulator = self.config['simulator']
        self.simulation_frequency = self.config['simulation_frequency']
        self.time_step_limit = self.config['time_step_limit']

    def _validate_config(self):
        """
        Validate the configuration dictionary. Raise ValueError for invalid configurations.
        """
        if 'env_name' not in self.config:
            raise ValueError("Environment name not specified in the configuration.")
        if 'observation_space' not in self.config:
            raise ValueError("Observation space details not specified in the configuration.")
        if 'action_space' not in self.config:
            raise ValueError("Action space details not specified in the configuration.")
        if 'reward_range' not in self.config:
            raise ValueError("Reward range not specified in the configuration.")
        if 'simulator' not in self.config:
            raise ValueError("Simulator details not specified in the configuration.")
        if 'simulation_frequency' not in self.config:
            raise ValueError("Simulation frequency not specified in the configuration.")
        if 'time_step_limit' not in self.config:
            raise ValueError("Time step limit not specified in the configuration.")

def load_config(config_file: str) -> Dict:
    """
    Load the configuration from a file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        Dict: Dictionary containing the configuration.
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    logger.info(f"Loading configuration from: {config_file}")
    return pd.read_json(config_file).to_dict()

def get_default_config() -> Dict:
    """
    Get the default configuration as a dictionary.

    Returns:
        Dict: Dictionary containing the default configuration.
    """
    default_config = {
        'env_name': 'Micromobility-v0',
        'state_size': 6,
        'action_size': 2,
        'learning_rate': 0.001,
        'buffer_size': 1000000,
        'batch_size': 256,
        'hidden_sizes': [64, 64],
        'tau': 0.005,
        'gamma': 0.99,
        'update_every': 1,
        'num_updates': 50,
        'target_update_interval': 500,
        'noise_scale': 0.1,
        'noise_reduction': 0.999,
        'random_seed': 42,
        'use_cuda': True,
        'simulator': 'CARLA',
        'simulation_frequency': 20,
        'time_step_limit': 1000,
        'observation_space': {'high': np.inf, 'low': -np.inf, 'dtype': 'float32'},
        'action_space': {'high': np.array([1.0, 1.0]), 'low': -np.array([1.0, 1.0]), 'dtype': 'float32'},
        'reward_range': [-float(1e6), float(1e6)]
    }
    return default_config

# Example usage
if __name__ == "__main__":
    config_dict = get_default_config()
    agent_config = AgentConfig(config_dict)
    env_config = EnvironmentConfig(config_dict)

    # Save configurations to files
    agent_config.save_config('agent_config.json')
    env_config.save_config('env_config.json')