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
	q =[]
	actions=[0, 1, 2]
	eps=0
	alpha=1
	discount=1
	def __init__(self,discount=1,eps=1):
		self.q= np.random.random([360*STEP_PER_DEG, 3])*1e-8
		#self.q= np.zeros([360*STEP_PER_DEG, 3])
		self.eps=eps
		self.discount=discount

	def policy(self, rel_wind_heading) -> int:
		return np.argmax(self.q[relangle_to_index(rel_wind_heading)]) if rd.random() > self.eps else rd.choice(self.actions)

	def step(self,r,s,sp,a):
		s=relangle_to_index(s)
		sp=relangle_to_index(sp)
		self.q[s,a]=( 1-self.alpha )* self.q[s,a] + self.alpha*( r + self.discount*max(self.q[sp] ))

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

def sigmoid(x):
	return np.where(x >= 0, 
			1 / (1 + np.exp(-x)), 
			np.exp(x) / (1 + np.exp(x)))


def grad_sigmoid(x):
        return sigmoid(x) * (1- sigmoid(x))

class LinSoftmax_agent:
	w=[[]]
	actions=[0, 1, 2]
	def __init__(self):
		#self.w=np.random.rand(3,5)
		self.w=np.ones((3,5))

	def predictor(self, rel_wind_heading, a):
		s=self.feature(rel_wind_heading)
		norm=np.exp(np.dot(self.w[0],s))+np.exp(np.dot(self.w[1],s))+np.exp(np.dot(self.w[2],s))
		return float(np.exp(np.dot(self.w[a],s))/ norm)

	def policy(self, rel_wind_heading) -> int:
		ran=rd.random()
		#print(ran,"          ",self.predictor(rounding_func(rel_wind_heading),0),"    ",self.predictor(rounding_func(rel_wind_heading),0) +self.predictor(rounding_func(rel_wind_heading),1))
		if ran<=self.predictor(rounding_func(rel_wind_heading),0):
			return 0
		elif ran<=self.predictor(rounding_func(rel_wind_heading),0) +self.predictor(rounding_func(rel_wind_heading),1):
			return 1
		else:
			return 2

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)

	def log_derivative(self,s,a):
		s=self.feature(s)
		norm=np.exp(np.dot(self.w[0],s))+np.exp(np.dot(self.w[1],s))+np.exp(np.dot(self.w[2],s))
		grad=s-s*float(np.exp(np.dot(self.w[a],s))/norm)
		ret=np.zeros([3,len(grad)])
		ret[a]=grad
		return ret
	
	def feature(self,s):
		s=s/180
		return np.array([1,s,s**2,s**3,s**4])

class LinExp_agent:
	w=[]
	actions=[0, 1, 2]
	def __init__(self):
		self.w=np.random.rand(3)

	def predictor(self, rel_wind_heading, a):
		return np.exp(np.dot(self.w,[1,rel_wind_heading,a]))/sum([np.exp(np.dot(self.w,[1,rel_wind_heading,ac])) for ac in self.actions])

	def policy(self, rel_wind_heading) -> int:
		ran=rd.random()
		if ran<=self.predictor(rounding_func(rel_wind_heading),0):
			return 0
		elif ran<=self.predictor(rounding_func(rel_wind_heading),0) +self.predictor(rounding_func(rel_wind_heading),1):
			return 1
		else:
			return 2

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)

	def log_derivative(self,s,a):
		return np.array([1,s,a])- sum([np.array([1,s,ac])*np.exp(np.dot(self.w,[1,s,ac])) for ac in self.actions])/sum([np.exp(np.dot(self.w,[1,s,ac])) for ac in self.actions])

class QuadExp_agent:
	w=[]
	actions=[0, 1, 2]
	def __init__(self):
		self.w=np.random.rand(4)

	def predictor(self, rel_wind_heading, a):
		return np.exp(np.dot(self.w,np.array([1,rel_wind_heading,rel_wind_heading**2,a])/362))\
			/sum([np.exp(np.dot(self.w,np.array([1,rel_wind_heading,rel_wind_heading**2,ac])/362)) for ac in self.actions]) 

	def policy(self, rel_wind_heading) -> int:
		ran=rd.random()
		if ran<=self.predictor(rounding_func(rel_wind_heading),1):
			return 1
		elif ran<=self.predictor(rounding_func(rel_wind_heading),1) +self.predictor(rounding_func(rel_wind_heading),2):
			return 2
		else:
			return 0

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)

	def log_derivative(self,s,a):
		s=s/360
		return np.array([1,s,s**2,a])/362- sum([np.array([1,s,s**2,a])/362*np.exp(np.dot(self.w,np.array([1,s,s**2,a])/362)) for ac in self.actions])/sum([np.exp(np.dot(self.w,np.array([1,s,s**2,a])/362)) for ac in self.actions])

class QuadExp_no_agent:
	w=[]
	actions=[0, 1, 2]
	def __init__(self):
		self.w=np.random.rand(4)

	def predictor(self, rel_wind_heading, a):
		return np.exp(np.dot(self.w,np.array([1,rel_wind_heading,rel_wind_heading**2,a])/362))

	def policy(self, rel_wind_heading) -> int:
		ran=rd.random()
		if ran<=self.predictor(rounding_func(rel_wind_heading),1):
			return 1
		elif ran<=self.predictor(rounding_func(rel_wind_heading),1) +self.predictor(rounding_func(rel_wind_heading),2):
			return 2
		else:
			return 0

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)

	def log_derivative(self,s,a):
		s=s/360
		return np.array([1,s,s**2,a])

class QuadExp_indip_agent:
	w0=[]
	w1=[]
	w2=[]
	actions=[0, 1, 2]
	def __init__(self):
		self.w1=np.random.rand(3)
		self.w2=np.random.rand(3)
		self.w0=np.random.rand(3)

	def predictor(self, rel_wind_heading, a):
		wind_norm=rel_wind_heading/360
		norm=np.exp(np.dot(self.w1,[1,wind_norm,wind_norm**2]))+np.exp(np.dot(self.w2,[1,wind_norm,wind_norm**2]))+np.exp(np.dot(self.w0,[1,wind_norm,wind_norm**2]))
		if a==0:
			return np.exp(np.dot(self.w0,[1,wind_norm,wind_norm**2]))/norm
		elif a==1:
			return np.exp(np.dot(self.w1,[1,wind_norm,wind_norm**2]))/norm
		else:
			return np.exp(np.dot(self.w2,[1,wind_norm,wind_norm**2]))/norm

	def policy(self, rel_wind_heading) -> int:
		ran=rd.random()
		if ran<=self.predictor(rounding_func(rel_wind_heading),1):
			return 1
		elif ran<=self.predictor(rounding_func(rel_wind_heading),1) +self.predictor(rounding_func(rel_wind_heading),2):
			return 2
		else:
			return 0

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)

	def log_derivative(self,s,a):
		s=s/360
		return np.array([1,s,s**2,a])


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

	def step_greedy(self, action):
		# Log the estimated wind
		self.rel_wind_heading_log[self.step_count] = wrap_to_m180_p180(self.wd.heading - self.wt.heading)

		# Log the true wind
		self.true_rel_wind_heading_log[self.step_count] = wrap_to_m180_p180(self.wd.heading - self.wt.true_heading)
		self.wd_heading_log[self.step_count] = self.wd.heading

		# Get action
		self.action_log[self.step_count] = action

		# Apply action and get power output
		self.power_output_log[self.step_count] = self.wt.step(self.wd.speed, self.wd.heading, self.action_log[self.step_count])

		# Generate new wind
		self.wd.step()

		# Like GymAI: S', R
		return wrap_to_m180_p180(self.wd.heading - self.wt.heading), self.power_output_log[self.step_count]

	
	def manual_step(self):
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
		return self.rel_wind_heading_log[self.step_count],wrap_to_m180_p180(self.wd.heading - self.wt.heading),\
			self.power_output_log[self.step_count],self.action_log[self.step_count]

	def run_simu(self):
		while self.step_count < self.max_steps:
			self.step()
			self.step_count += 1
	
	def get_initial_state(self):
		return wrap_to_m180_p180(self.wd.heading - self.wt.heading)

	def __str__(self):
		return str(self.__class__) + ": " + str(self.__dict__)

