import random

import numpy as np

from bereshit import Vector3


class Walk:
    def __init__(self, goal, scene):
        self.Agent = None
        self.goal = goal
        self.dt = 0
        self.size = 0
        self.beta = [0.5]
        self.passed = [0, 0]
        self.target_speed = 2.0
        self.body_vel = Vector3()
        self.lastDistance = 0
        self.speed = 1
        self.last_val = Vector3()
        self.total_dic = 0
        self.scene = scene

    def attach(self, parent):
        self.Agent = parent.get_component("Agent")
        self.trainer = self.Agent.trainer
        self.body_parts = parent.get_all_children_physics()
        self.body_parts = [item for item in self.body_parts if item not in self.scene]
        self.body_parts = [item for item in self.body_parts if item not in self.goal.children]

        self.servos = []
        self.feets = parent.search_by_name("feet")
        self.hip_bone = parent.search_by_name("hip_bone")[0]
        self.hip_bone.add_component(Legs(self.Agent, self, 0))
        self.feets[0].add_component(Legs(self.Agent, self, 2))
        self.feets[1].add_component(Legs(self.Agent, self, 1))

        for child in self.body_parts:
            if child.get_component("Servo"):
                self.servos.append(child)
            if child not in self.feets:
                child.add_component(BodyPart(self.Agent))

    def OnEpisodeBegin(self):
        self.parent.reset_to_default()
        self.goal.reset_to_default()
        self.size = 0.5
        self.next_goal()
        self.lastDistance = self.getDistance()
        self.total_dic = self.getDistance()

        self.set_body_val()

    def next_goal(self):
        self.hip_bone.Legs.Reset()
        self.feets[0].Legs.Reset()
        self.feets[1].Legs.Reset()

        self.size += 0.2
        pos = Vector3(random.uniform(-1, 1), 0, random.uniform(-1, 1)).normalized() * self.size
        pos.y = 14
        self.goal.local_position = pos
        # rot = Vector3(0, random.uniform(-180, 180), 0)
        # self.goal.local_rotation = rot


    def Update(self, dt):
        self.dt += dt
        self.set_body_val()
        self.add_observations()
        action = self.Agent.get_continuous_actions()
        self.move(action, dt)
        self.addRewardByVelocity()
        # self.addRewardByDistance()
        self.addRewardByPlacement()
        self.Agent.add_reward(-0.0005)


        self.printData()

    def addRewardByPlacement(self):
        foot1_grounded = self.feets[0].Collider.stay or self.feets[0].Collider.enter
        foot2_grounded = self.feets[1].Collider.stay or self.feets[1].Collider.enter

        # punish jumping too high
        # if self.hip_bone.position.y > 14.2:
        #     self.Agent.add_reward(-1.0)
        if self.hip_bone.position.y < 13:
            self.Agent.add_reward(-2)
            self.Agent.end_episode()


        # reward at least one foot touching ground
        if foot1_grounded or foot2_grounded:
            self.Agent.add_reward(0.0005)

        # upright reward
        # upright = self.hip_bone.up.dot(Vector3(0, 1, 0))
        # self.Agent.add_reward((upright) * 0.001)


        # # punish high vertical velocity
        # self.Agent.add_reward(-abs(self.hip_bone.Rigidbody.velocity.y) * 0.01)

        if self.hip_bone.Legs.finished + self.feets[0].Legs.finished + self.feets[1].Legs.finished == 3:
            self.next_goal()
            self.Agent.add_reward(2)
        # elif self.hip_bone.Legs.finished + self.feets[0].Legs.finished + self.feets[1].Legs.finished == 2:
        #     self.Agent.add_reward(0.005)


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

    def addRewardByDistance(self):
        dis = self.getDistance()
        progress = self.lastDistance - dis
        self.lastDistance = dis

        self.Agent.add_reward(progress * 0.02)



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
        hip_dis = (self.hip_bone.position - self.goal.children[0].position).magnitude()

        foot1_vec = self.feets[0].position - self.goal.children[1].position
        foot1_vec.y = 0
        foot1_dis = foot1_vec.magnitude()

        foot2_vec = self.feets[1].position - self.goal.children[2].position
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


        self.Agent.add_observation(self.goal.children[0].local_position)
        self.Agent.add_observation(self.goal.children[1].local_position)
        self.Agent.add_observation(self.goal.children[2].local_position)

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
                # average = self.Agent.trainer.get_average_reward()
                # if (self.passed[0] / self.passed[1] > 0.8) and average > 1.8:
                #     self.beta[0] += 0.5
                # self.dt = 0
                # print(self.beta[0], self.passed[0] / self.passed[1])
                # self.passed[0] = 0
                # self.passed[1] = 0.001

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
            self.Agent.add_reward(-2)
            self.Agent.end_episode()


class Legs:

    def __init__(self, agent, other, num):
        self.Agent = agent
        self.other = other
        self.num = num
        self.dt = 0
        self.reward = 0
        self.finished = False

    def OnCollisionEnter(self, Collision):
        if (Collision.other.parent in self.other.goal.children) and not self.finished:
            self.Agent.add_reward(0.5)
            self.finished = True

    def OnCollisionStay(self, Collision):
        if Collision.other.parent in self.other.goal.children:
            self.Agent.add_reward(0.004)
            self.finished = True

    def OnCollisionExit(self, Collision):
        if (Collision.other.parent in self.other.goal.children) and self.finished:
            self.Agent.add_reward(-0.5)
            self.finished = False

    def Reset(self):
        self.finished = False