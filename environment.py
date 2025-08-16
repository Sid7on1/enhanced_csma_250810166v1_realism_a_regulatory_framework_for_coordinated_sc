import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from enum import Enum
from threading import Lock

# Constants
VELOCITY_THRESHOLD = 5.0  # m/s
FLOW_THEORY_THRESHOLD = 0.5  # ratio

# Configuration
class EnvironmentConfig:
    def __init__(self, 
                 num_agents: int, 
                 num_steps: int, 
                 learning_rate: float, 
                 gamma: float, 
                 epsilon: float):
        """
        Environment configuration.

        Args:
        - num_agents (int): Number of agents in the environment.
        - num_steps (int): Number of steps in the environment.
        - learning_rate (float): Learning rate for the agents.
        - gamma (float): Discount factor for the agents.
        - epsilon (float): Exploration rate for the agents.
        """
        self.num_agents = num_agents
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon

# Exception classes
class EnvironmentError(Exception):
    """Base class for environment-related exceptions."""
    pass

class AgentNotFoundError(EnvironmentError):
    """Raised when an agent is not found in the environment."""
    pass

class InvalidActionError(EnvironmentError):
    """Raised when an invalid action is taken in the environment."""
    pass

# Data structures/models
class Agent:
    def __init__(self, 
                 id: int, 
                 position: np.ndarray, 
                 velocity: np.ndarray):
        """
        Agent model.

        Args:
        - id (int): Unique identifier for the agent.
        - position (np.ndarray): Position of the agent.
        - velocity (np.ndarray): Velocity of the agent.
        """
        self.id = id
        self.position = position
        self.velocity = velocity

class State:
    def __init__(self, 
                 agents: List[Agent], 
                 step: int):
        """
        Environment state.

        Args:
        - agents (List[Agent]): List of agents in the environment.
        - step (int): Current step in the environment.
        """
        self.agents = agents
        self.step = step

# Validation functions
def validate_agent(agent: Agent) -> None:
    """
    Validate an agent.

    Args:
    - agent (Agent): Agent to validate.

    Raises:
    - ValueError: If the agent is invalid.
    """
    if agent.id < 0:
        raise ValueError("Agent ID must be non-negative")
    if agent.position is None or agent.velocity is None:
        raise ValueError("Agent position and velocity must be set")

def validate_action(action: np.ndarray) -> None:
    """
    Validate an action.

    Args:
    - action (np.ndarray): Action to validate.

    Raises:
    - ValueError: If the action is invalid.
    """
    if action is None:
        raise ValueError("Action must be set")

# Utility methods
def calculate_distance(position1: np.ndarray, position2: np.ndarray) -> float:
    """
    Calculate the distance between two positions.

    Args:
    - position1 (np.ndarray): First position.
    - position2 (np.ndarray): Second position.

    Returns:
    - float: Distance between the two positions.
    """
    return np.linalg.norm(position1 - position2)

def calculate_velocity(velocity: np.ndarray) -> float:
    """
    Calculate the velocity magnitude.

    Args:
    - velocity (np.ndarray): Velocity vector.

    Returns:
    - float: Velocity magnitude.
    """
    return np.linalg.norm(velocity)

# Environment class
class Environment:
    def __init__(self, 
                 config: EnvironmentConfig):
        """
        Environment setup.

        Args:
        - config (EnvironmentConfig): Environment configuration.
        """
        self.config = config
        self.agents = []
        self.step = 0
        self.lock = Lock()

    def add_agent(self, 
                   agent: Agent) -> None:
        """
        Add an agent to the environment.

        Args:
        - agent (Agent): Agent to add.

        Raises:
        - AgentNotFoundError: If the agent is already in the environment.
        """
        with self.lock:
            if agent in self.agents:
                raise AgentNotFoundError("Agent already in the environment")
            self.agents.append(agent)

    def remove_agent(self, 
                      agent: Agent) -> None:
        """
        Remove an agent from the environment.

        Args:
        - agent (Agent): Agent to remove.

        Raises:
        - AgentNotFoundError: If the agent is not in the environment.
        """
        with self.lock:
            if agent not in self.agents:
                raise AgentNotFoundError("Agent not in the environment")
            self.agents.remove(agent)

    def take_action(self, 
                    agent: Agent, 
                    action: np.ndarray) -> None:
        """
        Take an action in the environment.

        Args:
        - agent (Agent): Agent taking the action.
        - action (np.ndarray): Action to take.

        Raises:
        - InvalidActionError: If the action is invalid.
        """
        with self.lock:
            validate_agent(agent)
            validate_action(action)
            # Update agent position and velocity
            agent.position += action
            agent.velocity = calculate_velocity(action)

    def get_state(self) -> State:
        """
        Get the current state of the environment.

        Returns:
        - State: Current state of the environment.
        """
        with self.lock:
            return State(self.agents, self.step)

    def step_environment(self) -> None:
        """
        Step the environment.
        """
        with self.lock:
            self.step += 1
            # Update agents
            for agent in self.agents:
                # Apply velocity threshold
                if calculate_velocity(agent.velocity) > VELOCITY_THRESHOLD:
                    agent.velocity *= 0.5
                # Apply flow theory threshold
                if calculate_distance(agent.position, np.zeros_like(agent.position)) > FLOW_THEORY_THRESHOLD:
                    agent.position *= 0.5

    def reset_environment(self) -> None:
        """
        Reset the environment.
        """
        with self.lock:
            self.agents = []
            self.step = 0

# Integration interfaces
class EnvironmentInterface(ABC):
    @abstractmethod
    def get_environment(self) -> Environment:
        """
        Get the environment.

        Returns:
        - Environment: Environment instance.
        """
        pass

class EnvironmentFactory(EnvironmentInterface):
    def __init__(self, 
                 config: EnvironmentConfig):
        """
        Environment factory.

        Args:
        - config (EnvironmentConfig): Environment configuration.
        """
        self.config = config

    def get_environment(self) -> Environment:
        """
        Get the environment.

        Returns:
        - Environment: Environment instance.
        """
        return Environment(self.config)

# Main function
def main():
    # Create environment configuration
    config = EnvironmentConfig(num_agents=10, num_steps=100, learning_rate=0.1, gamma=0.9, epsilon=0.1)

    # Create environment factory
    factory = EnvironmentFactory(config)

    # Get environment
    environment = factory.get_environment()

    # Add agents
    for i in range(config.num_agents):
        agent = Agent(id=i, position=np.random.rand(2), velocity=np.random.rand(2))
        environment.add_agent(agent)

    # Take actions
    for _ in range(config.num_steps):
        for agent in environment.agents:
            action = np.random.rand(2)
            environment.take_action(agent, action)

        # Step environment
        environment.step_environment()

    # Reset environment
    environment.reset_environment()

if __name__ == "__main__":
    main()