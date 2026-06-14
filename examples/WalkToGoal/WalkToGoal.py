import random

import numpy as np

from bereshit import Vector3


class Walk:
    def __init__(self, goal, size=20):
        self.Agent = None
        self.goal = goal
        self.dt = 0
        self.size = size
        self.beta = [1]
        self.passed = [0, 0]
        self.target_speed = 2.0
        self.expected_vel_vector_cache = Vector3()
        self.body_vel = Vector3()
        self.last_position = Vector3()
        self.lastDistance = 0
        self.speed = 1
        self.last_val = Vector3()
        self.total_dic = 0

    def attach(self, parent):
        self.Agent = parent.get_component("Agent")
        self.trainer = self.Agent.trainer
        self.body_parts = parent.get_all_children_physics()
        self.servos = []
        self.feets = parent.search_by_name("feet")
        self.hip_bone = parent.search_by_name("hip_bone")[0]
        self.hip_bone.add_component(Legs(self.Agent, self.passed))
        for child in self.body_parts:
            if child.get_component("Servo"):
                self.servos.append(child)
            if child not in self.feets:
                child.add_component(BodyPart(self.Agent, self.passed))

    def OnEpisodeBegin(self):
        self.parent.reset_to_default()
        self.goal.local_position = Vector3(-1 - self.beta[0], 13.8, 0)
        self.expected_vel = random.uniform(2, 5)
        self.last_position = self.get_average_position()
        self.lastDistance = self.getDistance()
        self.total_dic = self.getDistance()


        self.set_body_val()
        direction = (self.goal.position - self.last_position).normalized()
        self.last_val = self.body_vel.dot(direction)

    def Update(self, dt):


        self.dt += dt
        self.expected_vel_vector_cache = (self.goal.position - self.get_average_position()).normalized()
        self.set_body_val()
        self.add_observations()
        action = self.Agent.get_continuous_actions()
        self.move(action, dt)
        self.addRewardByVelocity()
        self.addRewardByDistance()
        self.addRewardByPlacement()
        self.printData()

    def addRewardByPlacement(self):
        foot1_grounded = self.feets[0].Collider.stay or self.feets[0].Collider.enter
        foot2_grounded = self.feets[1].Collider.stay or self.feets[1].Collider.enter

        # punish jumping too high
        if self.hip_bone.position.y > 14.2:
            self.Agent.add_reward(-1.0)

        # reward at least one foot touching ground
        if foot1_grounded or foot2_grounded:
            self.Agent.add_reward(0.0005)

        # upright reward
        upright = self.hip_bone.up.dot(Vector3(0, 1, 0))
        self.Agent.add_reward((upright) * 0.001)


        # punish high vertical velocity
        self.Agent.add_reward(-abs(self.hip_bone.Rigidbody.velocity.y) * 0.01)

    def addRewardByVelocity(self):
        # direction to goal on XZ plane
        direction = self.goal.position - self.hip_bone.position
        direction.y = 0

        if direction.magnitude() < 0.001:
            return

        direction = direction.normalized()

        # body velocity on XZ plane only
        vel = Vector3(self.body_vel.x, 0, self.body_vel.z)

        speed_toward_goal = vel.dot(direction)

        # reward only forward movement
        forward_reward = np.clip(speed_toward_goal / self.target_speed, -1, 1)

        self.Agent.add_reward(forward_reward * 0.005)

        # punish moving too fast / jumping style
        speed_error = abs(speed_toward_goal - self.target_speed)
        speed_reward = 1.0 - np.clip(speed_error / self.target_speed, 0, 1)

        self.Agent.add_reward(max(0, speed_reward * 0.003))


    def addRewardByDistance(self):
        dis = self.getDistance()
        progress = self.lastDistance - dis
        self.lastDistance = dis

        # only reward real progress
        self.Agent.add_reward(max(0, progress * 0.1))

        stat = (self.total_dic - dis) / self.total_dic

        self.Agent.add_reward(stat * 0.001)



    def set_body_val(self):
        val = Vector3()
        length = len(self.body_parts)
        for part in self.body_parts:
            val += part.Rigidbody.velocity
        self.body_vel = val / length

    def move(self, action, dt):
        for i, servo in enumerate(self.servos):
            servo.ServoController.move(action[i] * self.speed, dt)

    def getDistance(self):
        hip_dis = (self.hip_bone.position - self.goal.position).magnitude()

        foot1_vec = self.feets[0].position - self.goal.position
        foot1_vec.y = 0
        foot1_dis = foot1_vec.magnitude()

        foot2_vec = self.feets[1].position - self.goal.position
        foot2_vec.y = 0
        foot2_dis = foot2_vec.magnitude()

        return hip_dis * 0.5 + foot1_dis * 0.25 + foot2_dis * 0.25

    def add_observations(self):
        for body_part in self.body_parts:
            self.Agent.add_observation(body_part.local_position)
            self.Agent.add_observation(body_part.quaternion.normalized())
            self.Agent.add_observation(body_part.Rigidbody.velocity)
            self.Agent.add_observation(body_part.Rigidbody.angular_velocity)
        for servo in self.servos:
            self.Agent.add_observation(servo.ServoController.target_angle)


        self.Agent.add_observation(self.goal.local_position)
        self.Agent.add_observation(self.body_vel)
        self.Agent.add_observation(self.parent.findTheCenterOfMass())
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
                if self.passed[0] / self.passed[1] > 0.8:
                    self.beta[0] += 0.5
                self.dt = 0
                print(self.beta[0], self.passed[0] / self.passed[1])
                self.passed[0] = 0
                self.passed[1] = 0

    def spawn_Goal(self, beta, size):
        vec = beta + Vector3(random.random() * random.choice([-1, 1]), 0,
                             random.random() * random.choice([-1, 1])).normalized() * size / 2 * 1
        vec.y = 14
        return vec


class BodyPart:
    def __init__(self, agent, passed):
        self.Agent = agent
        self.passed = passed

    def OnCollisionEnter(self, Collision):
        if Collision.other.parent.get_component("Wall"):
            self.Agent.add_reward(-2)
            self.passed[1] += 1

            self.Agent.end_episode()


class Legs:

    def __init__(self, agent, passed):
        self.Agent = agent
        self.passed = passed
        self.dt = 0
        self.reward = 0

    def OnCollisionEnter(self, Collision):
        if Collision.other.parent.get_component("Goal"):
            self.Agent.add_reward(2)
            self.passed[0] += 1
            self.passed[1] += 1
            self.Agent.end_episode()


