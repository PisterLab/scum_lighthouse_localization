#(c) 2019 Brian Kilberg
from __future__ import division, print_function
import numpy as np
from py3dmath import Vec3, Rotation  # get from https://github.com/muellerlab/py3dmath
import copy
import math



class Estimator6Dof:

	#maps state to A matrix
	stateDict = {"x": 0,
			"y": 1,
			"z": 2,
			"vx": 3,
			"vy": 4,
			"vz": 5,
			"d0": 6,
			"d1": 7,
			"d2": 8}

	def __init__(self,mass, inertiaMatrix,pos = Vec3(0,0,0), vel = Vec3(0,0,0), att=Rotation.identity(),d = Vec3(0,0,0)):

		self._mass = mass
		self._inertia = inertiaMatrix

		self._state_m = State(pos,vel,d,att)
		self._state_p = State(pos,vel,d,att)
		self._Pp = None
		self._Pm = np.eye(9)*0.01
		self._stateHistP = []
		self._posNoise = [0.01,0.01,0.01]
		self._velNoise = [0.01,0.01,0.01]
		self._dNoise = [0.01,0.01,0.01]
		self._debug = False
		#state vars

	def debugLinearization(self):
		if self._debug == False:
			self._debug = True

	def linearizeDynamics(self,accImu, gyroImu, dt):
		#create linearized A matrix

		A = np.zeros((9,9))

		A_pos = np.eye(3)
		A_d = np.eye(3)

		#position to velocity model
		A_posvel = self._state_m.att.to_rotation_matrix()*dt
		#print(self._state_m.att.to_rotation_matrix)
		#print(A_posvel)

		#position to attitude model
		d0 = Vec3(1,0,0).to_cross_product_matrix()
		d1 = Vec3(0,1,0).to_cross_product_matrix()
		d2 = Vec3(0,0,1).to_cross_product_matrix()

		#dynamics are R*(I + |_del_|) * vel
		#print(self._state_m.vel.to_array())
		grad_d0 = self._state_m.att.to_rotation_matrix()*d0*self._state_m.vel.to_array() * dt
		grad_d1 = self._state_m.att.to_rotation_matrix()*d1*self._state_m.vel.to_array() * dt
		grad_d2 = self._state_m.att.to_rotation_matrix()*d2*self._state_m.vel.to_array() * dt


		A_pos_att = np.column_stack((grad_d0,grad_d1,grad_d2))

		#velocity to velocity = Rot(omega)_T
		A_velvel = gyroImu.to_cross_product_matrix().T

		#velocity from attitude error = -g*(-[del x])*R_T * e3
		grad_d0 = 9.81*d0*self._state_m.att.to_rotation_matrix().T*Vec3(0,0,1).to_array() * dt
		grad_d1 = 9.81*d1*self._state_m.att.to_rotation_matrix().T*Vec3(0,0,1).to_array() * dt
		grad_d2 = 9.81*d2*self._state_m.att.to_rotation_matrix().T*Vec3(0,0,1).to_array() * dt
		A_velatt = np.column_stack((grad_d0,grad_d1,grad_d2))

		#att error from att error. this is from the covariance update paper
		gyroCrossRod = gyroImu.to_cross_product_matrix()/2 * dt
		mat = np.eye(3)+gyroCrossRod.T + gyroCrossRod.T*gyroCrossRod.T/2

		A_attatt = mat

		#create A matrix
		A = np.block([[A_pos, A_posvel, A_pos_att],
					  [np.zeros((3,3)), A_velvel, A_velatt],
					  [np.zeros((3,3)), np.zeros((3,3)), A_attatt]])

		return A

	def kalmanPredict(self, accImu, gyroImu, dt):

		#linearize A matrix
		A = self.linearizeDynamics(accImu, gyroImu,dt)

		#process noise
		L = np.diag([dt, dt, dt, dt, dt, dt, dt, dt, dt])

		Q = np.diag(np.concatenate((self._posNoise, self._velNoise, self._dNoise)))

		#covariance update

		self._Pp = A * self._Pm * A.T + L * Q * L.T

		#print(A)
		#position prediction
		dposBody = self._state_m.vel * dt + Vec3(0,0,accImu[2]*dt*dt / 2.0)
		#self._state_p.pos = self._state_m.att * dposBody - Vec3(0,0,9.81*dt*dt / 2.0)
		self._state_p.pos = self._state_m.pos + self._state_m.att * dposBody - Vec3(0,0,9.81*dt*dt / 2.0)

		#velocity prediction: zacc - gyro x vel - g_body
		gyroCross = gyroImu.to_cross_product_matrix()
		self._state_p.vel = self._state_m.vel +  dt*(Vec3(0,0,accImu[2])-gyroCross*self._state_m.vel - self._state_m.att.inverse() * Vec3(0,0,9.81))
		#print(self._state_p.vel )
		#attitude update
		gyroRot = Rotation.from_rotation_vector(gyroImu*dt)
		self._state_p.att = self._state_m.att*gyroRot
		self._stateHistP.append(self._state_p)

		#compare linearized dynamics to actual dynamics
		#gyroRot = Rotation.from_rotation_vector(gyroImu*dt)
		#self._state_p.devectorize(A*self._state_m.getVector(),self._state_m.att*gyroRot)



	def kalmanUpdate(self, dt, lh_measurement):
		#no measurement update created yet
		if lh_measurement:
            self._state_m = copy.deepcopy(self._state_p)

		else:
			self._state_m = copy.deepcopy(self._state_p)
			self._Pm = self._Pp

class State:
	def __init__(self, pos, vel, d, att):
		self.pos = pos #position as Vec3
		self.vel = vel #velocity as Vec3
		self.d = d #attitude error as Vec3 (rodrigues params)
		self.att = att #reference attitude

	def getVector(self):
		return np.array([[self.pos[0], self.pos[1], self.pos[2], self.vel[0], self.vel[1], self.vel[2], self.d[0], self.d[1], self.d[2]]]).T

	def devectorize(self,state_vector,att):

		self.pos = Vec3(state_vector[0,0], state_vector[1,0], state_vector[2,0])
		self.vel = Vec3(state_vector[3,0], state_vector[4,0],state_vector[5,0])
		self.d = Vec3(state_vector[6,0], state_vector[7,0], state_vector[8,0])


		self.att = att
