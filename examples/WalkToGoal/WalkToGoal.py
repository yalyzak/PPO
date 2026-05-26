import random

import numpy as np

from bereshit import Vector3


class Walk:
    def __init__(self, goal, size=4):
        self.Agent = None
        self.speed = 10
        self.goal = goal
        self.dt = 0
        self.size = size
        self.beta = Vector3(-5,0,0)
        self.lastDistance = 0
        self.lastHeight = 0

    def attach(self, parent):
        self.Agent = parent.get_component("Agent")
        self.trainer = self.Agent.trainer
        self.body_parts = parent.get_all_children_physics()
        self.servos = []
        self.feets = parent.search_by_name("feet")
        self.hip_bone = parent.search_by_name("hip_bone")[0]
        for child in self.body_parts:
            if child.get_component("Servo"):
                self.servos.append(child)
            if child not in self.feets:
                child.add_component(BodyPart(self.Agent))
            else:
                child.add_component(Legs(self.Agent))

    def OnEpisodeBegin(self):
        self.parent.reset_to_default()
        self.goal.local_position = self.spawn_Goal(self.beta, self.size)
        self.lastDistance = self.getAverageDistance()
        self.lastHeight = self.getBestHeight()

    def Update(self, dt):
        self.dt += dt
        obs = self.get_observations()
        action = self.Agent.get_continuous_actions(obs)
        self.move(action, dt)
        self.addRewardByDistance()
        # self.addRewardByHeight()
        self.addRewardByTime(dt)
        self.printData()

    def addRewardByHeight(self):
        height = self.getBestHeight()
        reward = height - self.lastHeight
        self.lastHeight = height
        self.Agent.add_reward(reward)

    def addRewardByDistance(self):
        dis = self.getAverageDistance()
        if dis > 50:
            self.Agent.add_reward(-10)
            print("end")
            self.Agent.end_episode()
        reward = self.lastDistance - dis
        self.lastDistance = dis
        if reward > 0:
            self.Agent.add_reward(reward * 10)

    def addRewardByTime(self, dt):
        self.Agent.add_reward(dt * 0.0002)

    def move(self, action, dt):
        for i, servo in enumerate(self.servos):
            servo.ServoController.move(action[i], dt)

    def getAverageDistance(self):
        distance1 = (self.feets[0].position - self.goal.position).magnitude()
        distance2 = (self.feets[1].position - self.goal.position).magnitude()
        distance = (distance1 + distance2) / 2
        return distance

    def getBestHeight(self):
        foot_y = min(self.feets[0].position.y, self.feets[1].position.y)
        hip_y = self.hip_bone.position.y
        return hip_y - foot_y

    def get_observations(self):
        observations = np.empty((3 + 4 + 3 +3) * len(self.body_parts) + 3)
        i = 0
        for body_part in self.body_parts:
            pos = body_part.local_position.to_np()  # shape (3,)
            rot = body_part.quaternion.normalized().to_np()  # shape (4,)
            val = body_part.Rigidbody.velocity.normalized().to_np()  # shape (3,)
            ang_val = body_part.Rigidbody.angular_velocity.normalized().to_np()  # shape (3,)

            observations[i:i + 3] = pos
            i += 3

            observations[i:i + 4] = rot
            i += 4

            observations[i:i + 3] = val
            i += 3

            observations[i:i + 3] = ang_val
            i += 3
        observations[len(observations)-1] = self.goal.position.z
        observations[len(observations)-2] = self.goal.position.y
        observations[len(observations)-3] = self.goal.position.z

        return observations

    def printData(self):
        if self.dt > 0:
            stats = self.trainer.learn_if_ready()
            if stats is not None:
                print("PPO update:", stats)
                self.trainer.print_performance()
                self.dt = 0


    def spawn_Goal(self, beta, size):
        return beta + Vector3(random.random() * random.choice([-1, 1]), 0, random.random() * random.choice([-1, 1])).normalized() * size/2 * random.uniform(1,2)



class BodyPart:
    def __init__(self, agent):
        self.Agent = agent

    def OnCollisionEnter(self, Collision):
        if Collision.other.parent.get_component("Wall"):
            self.Agent.add_reward(-1)
            self.Agent.end_episode()


class Legs:

    def __init__(self, agent):
        self.Agent = agent

    def OnCollisionEnter(self, Collision):
        if Collision.other.parent.get_component("Goal"):
            self.Agent.add_reward(10)
            self.Agent.end_episode()

