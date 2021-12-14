import pybullet as p
import math as m
import numpy as np
import random as rand
import os
import pybullet_data
import math


class BallOnPlateEnv:
    def __init__(self):
        # Parameters

        # gym placeholdert
        self.observation_space_shape = [6]
        self.action_space_shape = [2]
        self.action_space_low = -1
        self.action_space_high = 1
from gym import spaces

class LabeledBox(spaces.Box):
    """
    Adds `labels` field to gym.spaces.Box to keep track of variable names.
    """
    def __init__(self, labels, **kwargs):
        super(LabeledBox, self).__init__(**kwargs)
        assert len(labels) == self.high.size
        self.labels = labels

class BallOnPlate:

    def __init__(self, showGUI=False, randomInitial=False):
        self.dt = 1 / 100.
        self.controlAngleLimit = 90
        self.plateSize = 2
        self.randomInitial = randomInitial

        self.intial_pos = np.array([0., 0.])

        self.ballPosition = self.intial_pos

        self.ballHeight = 0.8

        self.env = BallOnPlateEnv()

        # 连接物理引擎
        if showGUI:
            self.physId = p.connect(p.GUI)
        else:
            self.physId = p.connect(p.DIRECT)

        p.resetSimulation(physicsClientId=self.physId)
        # 重力
        p.setGravity(0, 0, -10, physicsClientId=self.physId)
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt, physicsClientId=self.physId)
        p.setTimeStep(1. / 240)
        ballRadius = 50 / 2 / 1000

        ballMass = 0.200
        ballInertia = 2. / 5 * ballMass * ballRadius * ballRadius

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane100.urdf", useMaximalCoordinates=True)

        self.plateId = p.loadURDF(os.path.dirname(os.path.realpath(__file__)) + '/plate.urdf',
                                  physicsClientId=self.physId)

        sphereCollisionShapeId = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=ballRadius,
                                                        physicsClientId=self.physId)


        self.ballId = p.createMultiBody(baseMass=ballMass, baseInertialFramePosition=[ballInertia] * 3,
                                        baseCollisionShapeIndex=sphereCollisionShapeId, baseVisualShapeIndex=-1,
                                        basePosition=[self.ballPosition[0], self.ballPosition[1], 0],
                                        physicsClientId=self.physId)

        self.angleTargets = [0, 0]
        self.platehight = 0.77918791



    def reset(self):

        self.time = 0

        # if self.randomInitial:
        #     self.intial_pos = np.array([((rand.random()-.5) * put_up), ((rand.random()-.5) * put_up)])

        self.ballHeight = 0.8
        self.ballPosition = self.intial_pos
        # Alpha, Beta

        #  posObj  reset the base of the object at the specified position in world space coordinates [X,Y,Z]
        #  ornObj  reset the base of the object at the specified orientation as world space quaternion
        p.resetBasePositionAndOrientation(bodyUniqueId=self.ballId,
                                          posObj=[self.ballPosition[0], self.ballPosition[1], self.ballHeight],
                                          ornObj=[0, 0, 0, 1], physicsClientId=self.physId)

        p.resetJointState(bodyUniqueId=self.plateId,
                          jointIndex=0, targetValue=0, targetVelocity=0,
                          physicsClientId=self.physId)

        p.resetJointState(bodyUniqueId=self.plateId, jointIndex=1, targetValue=0, targetVelocity=0,
                          physicsClientId=self.physId)

        ref_point = np.array([0., 0., 0.])
        state = self.state(ref_point)
        return state

    def state(self, ref_point):
        # 判断是否接触

        ballpos, ballorn = p.getBasePositionAndOrientation(self.ballId, physicsClientId=self.physId)

        ls = p.getLinkState(bodyUniqueId=self.plateId, linkIndex=1, physicsClientId=self.physId)


        platePos, plateOrn = ls[0], ls[1]

        plateorn1 = p.getEulerFromQuaternion(plateOrn)

        ballorn1 = p.getEulerFromQuaternion(ballorn)
        err = ballpos - ref_point[0:3]

        state = np.array(ballpos + plateorn1 )
        # print('2', state)
        return state

    def reward(self, state, done):
        # h 判断是否接触

        ## 控制reward=0
        if  self.is_contacted() == 0:
            reward = 0
        else:
            if done:
                reward = -10
            else:
                if state[0] == 0 and state[1] == 0 and state[2] - self.platehight < 1e-5:
                    r_ball = 10
                else:
                    r_ball = math.exp(-10*(state[0] ** 2 + state[1] ** 2 + (state[2] - self.platehight) ** 2))
                r_plate = -10*(state[3]**2+state[4]**2+state[5]**2)

                reward = r_ball + r_plate
                # print('2',  r_ball , r_plate )
        return reward

    def step(self, action, state):

        action = np.clip(action, self.env.action_space_low, self.env.action_space_high)
        self.angleTargets = np.array(action) * self.controlAngleLimit * m.pi / 180

        pos_prop = 0.01
        vel_prop = 1
        vel_max = 2
        force_max = 3

        p.setJointMotorControl2(bodyUniqueId=self.plateId, jointIndex=0, controlMode=p.POSITION_CONTROL,
                                positionGain=pos_prop, velocityGain=vel_prop, maxVelocity=vel_max,
                                targetPosition=self.angleTargets[0], force=force_max, physicsClientId=self.physId)

        p.setJointMotorControl2(bodyUniqueId=self.plateId, jointIndex=1, controlMode=p.POSITION_CONTROL,
                                positionGain=pos_prop, velocityGain=vel_prop, maxVelocity=vel_max,
                                targetPosition=self.angleTargets[1], force=force_max, physicsClientId=self.physId)

        p.stepSimulation(physicsClientId=self.physId)

        self.time += self.dt

        next_state = self.state(state)


        done = self._is_end(state)


        reward = self.reward(state, done)


        return next_state, reward, done, self.time

    def is_contacted(self):
        result0 = p.getContactPoints(self.ballId, self.plateId, physicsClientId=self.physId)
        # print('put_up', len(result0))
        return len(result0)


    def _is_end(self, state):

        done = False
        if state[2] < 0.05:
            done = True
        elif state[2] - 0.7 < 0 and self.is_contacted() == 0:
            done = True

        return done

    def close(self):
        # state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # self.step(np.array([0.0, 0.0]), state)

        p.disconnect(physicsClientId=self.physId)


   ### 打开力  ####
        # print('5', done, done1, done2)
        # p.enableJointForceTorqueSensor(bodyUniqueId=self.plateId,jointIndex=put_up, enableSensor=0)
        # p.getJointState(bodyUniqueId =self.plateId,jointIndex=put_up)
        # any(abs(self.ballPosition) > put_up.)