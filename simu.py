#!/usr/bin/env pythnon3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Paul Aubin
# Created Date: 2022/09/28
# ---------------------------------------------------------------------------
""" Setup a simulation of a wind turbine agent """
# ---------------------------------------------------------------------------
import numpy as np
from math_utils import wrap_to_m180_p180
from wind_turbine import Wind_turbine, Wind
import random as rd

STEP_PER_DEG = 1 # -> Corresponds to a disc of 0.333

def rounding_func(angle):
    return round(angle*STEP_PER_DEG)/STEP_PER_DEG

def angle_to_index(angle):
    return round(angle*STEP_PER_DEG)

def relangle_to_index(angle):
	return int(round((angle+179)*STEP_PER_DEG)/STEP_PER_DEG)

class Basic_agent:
	__threshold = 5 					# deg, corresponds to the wind deadzone in which no action is taken

	def __init__(self):
		pass

	def policy(self, rel_wind_heading) -> int:
		'''
		Define the policy of the agent
		Input  : relative wind heading between the wind and the wind turbine.
		Ouptut : an int corresponding to the selected action : 0 rotate clockwise, 1 do nothgin, 2 rotate trigo
		'''
		rel_wind_heading = wrap_to_m180_p180(rel_wind_heading)
		# If the relative angle to the wind is low do nothing
		if np.abs(rel_wind_heading) - self.__threshold < 0:
			return 1
		# Else follow the wind
		else:
			return np.sign(rel_wind_heading) + 1

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)

class Tracking_agent:
	"""Custom agent designed to continually track and adjust rotation based on wind orientation
	"""
	def __init__(self):
		pass

	def policy(self, rel_wind_heading) -> int:
		'''
		Define the policy of the agent
		Input  : relative wind heading between the wind and the wind turbine.
		Ouptut : an int corresponding to the selected action : 0 rotate clockwise, 1 do nothgin, 2 rotate trigo
		'''
		rel_wind_heading = wrap_to_m180_p180(rel_wind_heading)
		return np.sign(rel_wind_heading) + 1

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)

class Random_agent:
	def __init__(self):
		pass

	def policy(self, rel_wind_heading) -> int:
		'''
		Define the policy of the agent, as a random agent it selects a random action given a uniform probability distribution
		Ouptut : an int corresponding to the selected action : 0 rotate clockwise, 1 do nothgin, 2 rotate trigo
		'''
		return np.random.choice([0, 1, 2])

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)


class Epsgreedy_agent:
	def __init__(self):
		pass

	def policy(self, rel_wind_heading, Q, eps) -> int:
		return np.argmax(Q[relangle_to_index(rel_wind_heading)]) if rd.random() > eps else rd.choice([0, 1, 2])

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)


# An agent to be completed
class Custom_agent:
	def __init__(self):
		pass

	def policy(self, rel_wind_heading) -> int:
		pass

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)

class Mapping_agent:
	"""Agent that can be initialized with a fixed mapping state to action (= a policy).
	If not specified, the initial policy is a random one."""
	policy_map = None

	def __init__(self, mapping=None):
		if mapping == None:
			self.policy_map = [rd.choice([0, 1, 2]) for _ in range(360*STEP_PER_DEG)]
		else:
			self.policy_map = mapping

	def policy(self, rel_wind_heading) -> int:
		return self.policy_map[angle_to_index(rel_wind_heading)]

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)


class Simu:
	power_output_log = [] 			# MW
	action_log = []
	rel_wind_heading_log = []		# deg
	true_rel_wind_heading_log = [] 	# deg
	wd_heading_log = [] 			# deg
	step_count = 0

	def __init__(self, agent=None, wind_model=None, wind_turbine_model=None, max_steps=None):
		self.wd = Wind(10, 0, 1, 'OU') if wind_model is None else wind_model
		self.wt = Wind_turbine(0, False) if wind_turbine_model is None else wind_turbine_model
		self.agent = Basic_agent() if agent is None else agent
		self.max_steps = 24*3600 if max_steps is None else max_steps 
		self.power_output_log = self.max_steps * [0]
		self.action_log = self.max_steps * [1]
		self.rel_wind_heading_log = self.max_steps * [0]
		self.true_rel_wind_heading_log = self.max_steps * [0]
		self.wd_heading_log = self.max_steps * [0]

	def step(self):
		# Log the estimated wind
		self.rel_wind_heading_log[self.step_count] = wrap_to_m180_p180(self.wd.heading - self.wt.heading)

		# Log the true wind
		self.true_rel_wind_heading_log[self.step_count] = wrap_to_m180_p180(self.wd.heading - self.wt.true_heading)
		self.wd_heading_log[self.step_count] = self.wd.heading

		# Get action
		self.action_log[self.step_count] = self.agent.policy(self.rel_wind_heading_log[self.step_count])

		# Apply action and get power output
		self.power_output_log[self.step_count] = self.wt.step(self.wd.speed, self.wd.heading, self.action_log[self.step_count])

		# Generate new wind
		self.wd.step()

	def step_enhanced(self, action):
		"""Step function returning the new state and the reward resulting from taking
		the action passed in argument to this function."""
		# Log the estimated wind (logging St)
		self.rel_wind_heading_log[self.step_count] = wrap_to_m180_p180(self.wd.heading - self.wt.heading)

		# Log the true wind
		self.true_rel_wind_heading_log[self.step_count] = wrap_to_m180_p180(self.wd.heading - self.wt.true_heading)
		self.wd_heading_log[self.step_count] = self.wd.heading

		# Get action (logging At)
		self.action_log[self.step_count] = action

		# Apply action and get power output (Logging Rt+1)
		self.power_output_log[self.step_count] = self.wt.step(self.wd.speed, self.wd.heading, self.action_log[self.step_count])

		# Generate new wind
		self.wd.step()

		# GymAI step function return format: St+1, Rt+1
		return wrap_to_m180_p180(self.wd.heading - self.wt.heading), self.power_output_log[self.step_count]

	
	def manual_step(self, action):
		"""Manually apply a step of the environnment
		and returns new state and reward"""
		if self.step_count < self.max_steps:
			stp, rt = self.step_enhanced(action)
			
			self.step_count += 1

			return stp, rt, self.step_count >= self.max_steps
		else:
			raise Exception("Max steps reached")

	def run_simu(self):
		while self.step_count < self.max_steps:
			self.step()
			self.step_count += 1
	
	def get_initial_state(self):
		return wrap_to_m180_p180(self.wd.heading - self.wt.heading)

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)

