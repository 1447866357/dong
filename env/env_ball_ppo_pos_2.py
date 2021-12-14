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
        self.observation_space_shape = [6]
        self.action_space_shape = [2]
        self.action_space_low = -1
        self.action_space_high = 1


class BallOnPlate:

    def __init__(self, showGUI=False, randomInitial=False):
        self.dt = 1 / 100.
        self.controlAngleLimit = 45
        self.plateSize = 2
        self.randomInitial = randomInitial

        self.intial_pos = np.array([0., 0.])

        self.ballPosition = self.intial_pos



        self.env_space = BallOnPlateEnv()

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
        self.platehight = 0.7764151

        # self.reset()


    def reset(self):

        self.time = 0

        # if self.randomInitial:
        #     self.intial_pos = np.array([((rand.random()-0.5) * 0.2), ((rand.random()-0.5) * 0.2)])


        self.ballPosition = self.intial_pos
        self.ballHeight = 0.8
        self.angleTargets = [0, 0]
        p.resetBasePositionAndOrientation(bodyUniqueId=self.ballId,
                                          posObj=[self.ballPosition[0], self.ballPosition[1], self.ballHeight],
                                          ornObj=[0, 0, 0, 1], physicsClientId=self.physId)

        p.resetJointState(bodyUniqueId=self.plateId,
                          jointIndex=0,
                          targetValue=0,
                          targetVelocity=0,
                          physicsClientId=self.physId)

        p.resetJointState(bodyUniqueId=self.plateId,
                          jointIndex=1,
                          targetValue=0,
                          targetVelocity=0,
                          physicsClientId=self.physId)

        ref_point = np.array([0., 0., 0.])


        while self.is_contacted() == 0:
            # print('2')
            p.stepSimulation(physicsClientId=self.physId)

        state = self.state(ref_point)

        return state

    def state(self, ref_point):
        # 判断是否接触

        ballpos, ballorn = p.getBasePositionAndOrientation(self.ballId, physicsClientId=self.physId)

        ls = p.getLinkState(bodyUniqueId=self.plateId,
                            linkIndex=1,
                            physicsClientId=self.physId)


        # platePos, plateOrn = ls[0], ls[1]

        # plateorn1 = p.getEulerFromQuaternion(plateOrn)
        #
        # ballorn1 = p.getEulerFromQuaternion(ballorn)
        # err = ballpos - ref_point[0:3]

        state = np.array(ballpos + ballpos)
        self.ballHeight = ballpos[2]
        return state



    def reward(self, state, done):
        ls = p.getLinkState(self.plateId, linkIndex=1, computeLinkVelocity=1, physicsClientId=self.physId)
        ballpos, ballorn = p.getBasePositionAndOrientation(self.ballId, physicsClientId=self.physId)
        platePos, plateOrn = ls[0], ls[1]
        plateorn1 = p.getEulerFromQuaternion(plateOrn)

        reward1 = 0
        reward2 = 0
        r_ball = 0
        if self.is_contacted() == 0:

            reward1 = 1
        if done:
            reward2 = -10

        # if ballpos[0] == 0 and ballpos[1] == 0:
        #     r_ball = 10
        # # else:
        #     r_ball = math.exp(-0.001*(ballpos[0] ** 2 + ballpos[1] ** 2))
        #
        # r_plate = -0.1 * (plateorn1[0] ** 2 + plateorn1[0] ** 2 + plateorn1[2] ** 2)
        # print('2',   r_ball , r_plate)
        reward = reward1 + reward2 + r_ball
        # print('3', reward )
        return reward


    def step(self, action, state):

        action = np.clip(action, self.env_space.action_space_low, self.env_space.action_space_high)
        self.angleTargets = np.array(action) * self.controlAngleLimit * m.pi / 180
        # print('put_up', action, self.angleTargets[0], self.angleTargets[put_up], self.angleTargets*3 )
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


        done = self._is_end(next_state)


        reward = self.reward(next_state, done)


        return next_state, reward, done, self.time

    def is_contacted(self):
        result0 = p.getContactPoints(self.ballId,
                                     self.plateId,
                                     linkIndexB=1,
                                     physicsClientId=self.physId)

        return len(result0)


    def _is_end(self, state):

        done = False
        if self.ballHeight < 0.1:
            done = True
        elif self.ballHeight - 0.6 < 0 and self.is_contacted() == 0:
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