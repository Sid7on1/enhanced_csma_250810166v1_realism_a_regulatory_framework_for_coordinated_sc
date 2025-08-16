import logging
import threading
from typing import Dict, List
import numpy as np
import torch
import pandas as pd
from enum import Enum

# Define constants
VELOCITY_THRESHOLD = 5.0  # velocity threshold from the paper
FLOW_THEORY_CONSTANT = 0.1  # flow theory constant from the paper

# Define exception classes
class MultiAgentCommException(Exception):
    """Base exception class for multi-agent communication"""
    pass

class AgentNotFoundException(MultiAgentCommException):
    """Exception raised when an agent is not found"""
    pass

class InvalidMessageException(MultiAgentCommException):
    """Exception raised when an invalid message is received"""
    pass

# Define logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# Define the Agent class
class Agent:
    """Represents an agent in the multi-agent system"""
    def __init__(self, agent_id: str, velocity: float):
        """
        Initializes an Agent object.

        Args:
        - agent_id (str): Unique identifier for the agent
        - velocity (float): Velocity of the agent
        """
        self.agent_id = agent_id
        self.velocity = velocity

    def __str__(self):
        return f"Agent {self.agent_id} with velocity {self.velocity}"

# Define the MultiAgentComm class
class MultiAgentComm:
    """Manages communication between multiple agents"""
    def __init__(self):
        """
        Initializes a MultiAgentComm object.
        """
        self.agents: Dict[str, Agent] = {}  # dictionary of agents
        self.lock = threading.Lock()  # lock for thread safety

    def add_agent(self, agent: Agent):
        """
        Adds an agent to the multi-agent system.

        Args:
        - agent (Agent): Agent to add

        Raises:
        - AgentNotFoundException: If the agent is already added
        """
        with self.lock:
            if agent.agent_id in self.agents:
                raise AgentNotFoundException(f"Agent {agent.agent_id} already exists")
            self.agents[agent.agent_id] = agent
            logging.info(f"Added agent {agent}")

    def remove_agent(self, agent_id: str):
        """
        Removes an agent from the multi-agent system.

        Args:
        - agent_id (str): ID of the agent to remove

        Raises:
        - AgentNotFoundException: If the agent is not found
        """
        with self.lock:
            if agent_id not in self.agents:
                raise AgentNotFoundException(f"Agent {agent_id} not found")
            del self.agents[agent_id]
            logging.info(f"Removed agent {agent_id}")

    def send_message(self, sender_id: str, recipient_id: str, message: str):
        """
        Sends a message from one agent to another.

        Args:
        - sender_id (str): ID of the sender agent
        - recipient_id (str): ID of the recipient agent
        - message (str): Message to send

        Raises:
        - AgentNotFoundException: If the sender or recipient agent is not found
        - InvalidMessageException: If the message is invalid
        """
        with self.lock:
            if sender_id not in self.agents:
                raise AgentNotFoundException(f"Sender agent {sender_id} not found")
            if recipient_id not in self.agents:
                raise AgentNotFoundException(f"Recipient agent {recipient_id} not found")
            if not message:
                raise InvalidMessageException("Invalid message")
            logging.info(f"Sent message from {sender_id} to {recipient_id}: {message}")

    def apply_velocity_threshold(self, agent_id: str):
        """
        Applies the velocity threshold to an agent.

        Args:
        - agent_id (str): ID of the agent

        Raises:
        - AgentNotFoundException: If the agent is not found
        """
        with self.lock:
            if agent_id not in self.agents:
                raise AgentNotFoundException(f"Agent {agent_id} not found")
            agent = self.agents[agent_id]
            if agent.velocity > VELOCITY_THRESHOLD:
                logging.info(f"Applied velocity threshold to agent {agent_id}")
                # Apply the velocity threshold formula from the paper
                agent.velocity = VELOCITY_THRESHOLD - (agent.velocity - VELOCITY_THRESHOLD) * FLOW_THEORY_CONSTANT

    def get_agent_velocity(self, agent_id: str) -> float:
        """
        Gets the velocity of an agent.

        Args:
        - agent_id (str): ID of the agent

        Returns:
        - float: Velocity of the agent

        Raises:
        - AgentNotFoundException: If the agent is not found
        """
        with self.lock:
            if agent_id not in self.agents:
                raise AgentNotFoundException(f"Agent {agent_id} not found")
            return self.agents[agent_id].velocity

# Define the main function
def main():
    # Create a multi-agent communication object
    multi_agent_comm = MultiAgentComm()

    # Create agents
    agent1 = Agent("agent1", 10.0)
    agent2 = Agent("agent2", 5.0)

    # Add agents to the multi-agent system
    multi_agent_comm.add_agent(agent1)
    multi_agent_comm.add_agent(agent2)

    # Send a message from one agent to another
    multi_agent_comm.send_message("agent1", "agent2", "Hello, agent2!")

    # Apply the velocity threshold to an agent
    multi_agent_comm.apply_velocity_threshold("agent1")

    # Get the velocity of an agent
    velocity = multi_agent_comm.get_agent_velocity("agent1")
    logging.info(f"Velocity of agent1: {velocity}")

    # Remove an agent from the multi-agent system
    multi_agent_comm.remove_agent("agent2")

if __name__ == "__main__":
    main()