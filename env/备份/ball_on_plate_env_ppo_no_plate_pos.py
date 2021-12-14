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
        self.observation_space_shape = [2]
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
        self.controlAngleLimit = 20

        self.randomInitial = randomInitial

        self.intial_pos = np.array([0., 0.])
        self.PLATE_SIDE_SZ = 1
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
        p.setGravity(0, 0, -9.80665, physicsClientId=self.physId)
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt, physicsClientId=self.physId)
        p.setTimeStep(1. / 240)
        import time
        # time.sleep(10)
        ballRadius = 52.2 / 2 / 1000

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
        self.platehight = 0.776
        self.raw_pid_rates = [0.8314504111768981, 1.1925901500923737, 0.016085473303319286]

        self.reset()

    def reset(self):

        self.time = 0

        if self.randomInitial:
            self.intial_pos = np.array([((rand.random()-.5) * 1), ((rand.random()-.5) * 1)])

        self.ballHeight = 0.8
        self.ballPosition = self.intial_pos * self.PLATE_SIDE_SZ
        # Alpha, Beta

        a = p.getNumJoints(self.plateId)
        b = p.getJointInfo(self.plateId, 0)
        c = p.getJointInfo(self.plateId, 1)
        # print('put_up', a)
        # print('2', b)
        # print('3', c)
        #  posObj  reset the base of the object at the specified position in world space coordinates [X,Y,Z]
        #  ornObj  reset the base of the object at the specified orientation as world space quaternion
        p.resetBasePositionAndOrientation(bodyUniqueId=self.ballId,
                                          posObj=[self.ballPosition[0], self.ballPosition[1], self.ballHeight],
                                          ornObj=[0, 0, 0, 1], physicsClientId=self.physId)

        p.resetJointState(self.plateId,
                          jointIndex=0,
                          targetValue=0,
                          targetVelocity=0,
                          physicsClientId=self.physId)

        p.resetJointState(self.plateId,
                          jointIndex=1,
                          targetValue=0,
                          targetVelocity=0,
                          physicsClientId=self.physId)

        ref_point = np.array([0., 0., 0.])
        state = self.state(ref_point)
        while not self.is_contacted():
            p.stepSimulation(physicsClientId=self.physId)

        return state



    def state(self, state):
        # 判断是否接触

        ballpos, ballorn = p.getBasePositionAndOrientation(self.ballId, physicsClientId=self.physId)
        ballvel = p.getBaseVelocity(self.ballId, physicsClientId=self.physId)

        platepos, plateorn = p.getBasePositionAndOrientation(self.plateId, physicsClientId=self.physId)
        # print('2', platepos, plateorn)

        ls = p.getLinkState(self.plateId,  linkIndex=1,computeLinkVelocity = 1,  physicsClientId=self.physId)


        platePos, plateOrn = ls[0], ls[1]
        platevel = p.getBaseVelocity(self.plateId, physicsClientId=self.physId)

        invPlatePos, invPlateOrn = p.invertTransform(platePos,
                                                     plateOrn,
                                                     physicsClientId=self.physId)
        ballPosOnPlate, ballOrnOnPlate = p.multiplyTransforms(invPlatePos,
                                                              invPlateOrn,
                                                              ballpos,
                                                              ballorn,
                                                              physicsClientId=self.physId)
        # print('5',ballvel, ballvel2)

        platePos, plateOrn, platevel = ls[0], ls[1], ls[7]

        ballPosOnPlate = np.array(ballPosOnPlate)

        self.ballPosition = ballPosOnPlate[0:2] / self.PLATE_SIDE_SZ
        self.ballHeight = ballpos[2]



        next_state = np.array(self.ballPosition)
        self.ballHeight = ballpos[2]
        # print()
        return next_state

    def reward(self, state, done):
        ls = p.getLinkState(bodyUniqueId=self.plateId, computeLinkVelocity=1, linkIndex=1, physicsClientId=self.physId)

        platePos, plateOrn = ls[0], ls[1]
        plateorn1 = p.getEulerFromQuaternion(plateOrn)

        ## 控制reward=0
        if self.is_contacted() == 0:
            reward = 0
        else:

            if done:
                reward = -10
            else:
                if state[0] == 0 and state[1] == 0 and self.platehight - self.ballHeight < 1e-2:
                    r_ball = 10
                    # print('aa')
                else:
                    r_ball = 10*math.exp(-(state[0] ** 2 + state[1] ** 2 ))
                r_plate = -100*(plateorn1[0] ** 2 + plateorn1[0] ** 2 + plateorn1[2] ** 2)

                reward = r_ball + r_plate
                # print('2',  r_ball , r_plate )
        return reward

    def step(self, action, state):

        if self.is_contacted() == 0:
            action = [0, 0]
        else:
            action = np.clip(action, self.env.action_space_low, self.env.action_space_high)
        self.angleTargets = np.array(action) * self.controlAngleLimit* np.pi/180
        # print('put_up', action, self.angleTargets[0], self.angleTargets[put_up], )


        # Bullet params
        MAX_FORCE = 3
        MAX_VELOCITY = 2
        VELOCITY_GAIN = 1
        POSITION_GAIN = 0.01

        p.setJointMotorControl2(bodyUniqueId=self.plateId,
                                jointIndex=0,
                                controlMode=p.POSITION_CONTROL,
                                positionGain= POSITION_GAIN,
                                velocityGain= VELOCITY_GAIN,
                                maxVelocity= MAX_VELOCITY,
                                targetPosition=self.angleTargets[0],
                                force= MAX_FORCE,
                                physicsClientId=self.physId)

        p.setJointMotorControl2(bodyUniqueId=self.plateId,
                                jointIndex=1,
                                controlMode=p.POSITION_CONTROL,
                                positionGain= POSITION_GAIN,
                                velocityGain= VELOCITY_GAIN,
                                maxVelocity= MAX_VELOCITY,
                                targetPosition= self.angleTargets[1],
                                force= MAX_FORCE,
                                physicsClientId= self.physId)

        # while not self.is_contacted():
        #     p.stepSimulation(physicsClientId=self.physId)
        p.stepSimulation(physicsClientId=self.physId)

        self.time += self.dt

        next_state = self.state(state)

        done = self._is_end(state)

        reward = self.reward(state, done)

        return next_state, reward, done, self.time

    def is_contacted(self):
        result0 = p.getContactPoints(self.ballId, self.plateId, physicsClientId=self.physId)
        return len(result0)

    def _is_end(self, state):
        done = False
        if self.ballHeight < 0.1:
            done = True
        elif self.ballHeight - 0.3 < 0 and self.is_contacted() == 0:
            done = True
        return done

    def close(self):
        p.disconnect(physicsClientId=self.physId)

### 打开力  ####
# print('5', done, done1, done2)
# p.enableJointForceTorqueSensor(bodyUniqueId=self.plateId,jointIndex=put_up, enableSensor=0)
# p.getJointState(bodyUniqueId =self.plateId,jointIndex=put_up)
# any(abs(self.ballPosition) > put_up.)