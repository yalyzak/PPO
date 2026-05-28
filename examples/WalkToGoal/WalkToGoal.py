import random

import numpy as np

from bereshit import Vector3


class Walk:
    def __init__(self, goal, size=20):
        self.Agent = None
        self.speed = 10
        self.goal = goal
        self.dt = 0
        self.size = size
        self.beta = Vector3(-1, 0, 1)
        self.expected_vel = 2
        self.expected_vel_vector_cache = Vector3()
        self.body_vel = Vector3()
        self.last_position = Vector3()
        self.lastDistance = 0

    def attach(self, parent):
        self.Agent = parent.get_component("Agent")
        self.trainer = self.Agent.trainer
        self.body_parts = parent.get_all_children_physics()
        self.servos = []
        self.feets = parent.search_by_name("feet")
        self.hip_bone = parent.search_by_name("hip_bone")[0]
        self.hip_bone.add_component(Legs(self.Agent))
        for child in self.body_parts:
            if child.get_component("Servo"):
                self.servos.append(child)
            if child not in self.feets:
                child.add_component(BodyPart(self.Agent))


    def OnEpisodeBegin(self):
        self.parent.reset_to_default()
        self.goal.local_position = self.spawn_Goal(self.beta, self.size)
        self.expected_vel = random.uniform(2, 5)
        self.last_position = self.get_average_position()
        self.lastDistance = self.getAverageDistance()


    def Update(self, dt):
        self.dt += dt
        self.expected_vel_vector_cache = (self.goal.position - self.get_average_position()).normalized()
        self.set_body_val()
        self.add_observations()
        action = self.Agent.get_continuous_actions()
        self.move(action, dt)
        self.addRewardByVelocity()
        self.addRewardByDistance()
        self.printData()

    def addRewardByVelocity(self):
        current = self.get_average_position()
        direction = (self.goal.position - current).normalized()
        progress = (current - self.last_position).dot(direction)
        self.Agent.add_reward(progress * 0.01)
        self.last_position = current

    def addRewardByDistance(self):
        dis = self.getAverageDistance()
        reward = self.lastDistance - dis
        self.lastDistance = dis
        self.Agent.add_reward(reward * 0.2)

    def set_body_val(self):
        val = Vector3()
        length = len(self.body_parts)
        for part in self.body_parts:
            val += part.Rigidbody.velocity

        self.body_vel = val / length


    def move(self, action, dt):
        for i, servo in enumerate(self.servos):
            servo.ServoController.move(action[i], dt)

    def getAverageDistance(self):
        distance1 = (self.feets[0].position - self.goal.position).magnitude()
        distance2 = (self.feets[1].position - self.goal.position).magnitude()
        distance = (distance1 + distance2) / 2
        return distance

    def add_observations(self):
        for body_part in self.body_parts:
            self.Agent.add_observation(body_part.local_position)
            self.Agent.add_observation(body_part.quaternion.normalized())
            self.Agent.add_observation(body_part.Rigidbody.velocity)
            self.Agent.add_observation(body_part.Rigidbody.angular_velocity)

        self.Agent.add_observation(self.goal.local_position)
        self.Agent.add_observation(self.expected_vel_vector_cache * self.expected_vel)
        self.Agent.add_observation(self.body_vel)
        self.Agent.add_observation(self.feets[0].Collider.stay)
        self.Agent.add_observation(self.feets[1].Collider.stay)

    def get_average_position(self):
        average = Vector3()
        length = len(self.body_parts)
        for part in self.body_parts:
            average += part.position
        return average / length

    def printData(self):
        if self.dt > 0:
            stats = self.trainer.learn_if_ready()
            if stats is not None:
                print("PPO update:", stats)
                self.trainer.print_performance()
                self.dt = 0

    def spawn_Goal(self, beta, size):
        vec = beta + Vector3(random.random() * random.choice([-1, 1]), 0,
                              random.random() * random.choice([-1, 1])).normalized() * size / 2 * 1
        vec.y = 14
        return vec


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
