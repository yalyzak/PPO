import random

import numpy as np

from bereshit import Vector3, Component
from bereshit.addons.PPO import Agent


class Walk(Agent):
    def __init__(self, goal, scene):
        super(Walk, self).__init__()
        self.goal = goal
        self.dt = 0
        self.size = 1
        self.passed = [0, 0]
        self.fall = True
        self.target_speed = 2.0
        self.body_vel = Vector3()
        self.last_val = Vector3()
        self.lastDistance = 0
        self.speed = 1
        self.total_dic = 0
        self.scene = scene

    def attach(self, parent):
        self.body_parts = parent.get_all_children_physics()
        self.body_parts = [item for item in self.body_parts if item not in self.scene]
        self.body_parts = [item for item in self.body_parts if item not in self.goal.children]

        self.servos = []
        self.feets = parent.search_by_name("feet")
        self.hip_bone = parent.search_by_name("hip_bone")[0]
        self.hip_bone.add_component(Legs(self, self))

        for child in self.body_parts:
            if child.get_component("Servo"):
                self.servos.append(child)
            if child not in self.feets:
                child.add_component(BodyPart(self, self))

    def OnEpisodeBegin(self):
        self.parent.reset_to_default()
        self.goal.reset_to_default()
        self.size = 10
        self.next_goal()
        self.lastDistance = self.getDistance()
        self.total_dic = self.getDistance()

        self.set_body_val()

    def next_goal(self):
        self.hip_bone.Legs.Reset()
        # self.feets[0].Legs.Reset()
        # self.feets[1].Legs.Reset()

        self.size += 0.5
        pos = Vector3(random.uniform(-1, 1), 0, random.uniform(-1, 1)).normalized() * self.size
        pos.y = 14
        pos.z += 1.5
        self.goal.local_position = pos

        # rot = Vector3(0, random.uniform(-180, 180), 0)
        # self.goal.local_rotation = rot


    def Update(self, dt):
        self.dt += dt
        self.set_body_val()
        self.add_observations()
        action = self.get_continuous_actions()
        self.move(action, dt)
        # self.addRewardByVelocity()
        self.addRewardByDistance()
        self.addRewardByPlacement()
        self.printData()
        # self.Agent.add_reward(-0.0000005)

    def addRewardByPlacement(self):
        foot1_grounded = self.feets[0].Collider.stay or self.feets[0].Collider.enter
        foot2_grounded = self.feets[1].Collider.stay or self.feets[1].Collider.enter

        # punish jumping too high
        # if self.hip_bone.position.y > 14.2:
        #     self.Agent.add_reward(-1.0)
        if self.hip_bone.position.y < 13:
            self.add_reward(-0.5)
            self.fall = True
            self.end_episode()
        # up_direction = self.goal.position.y - self.hip_bone.position.y
        #
        # if up_direction > 0.009:
        #     self.Agent.add_reward(-2)
        #     self.fall = True
        #     self.Agent.end_episode()



        # reward at least one foot touching ground
        # if foot1_grounded or foot2_grounded:
        #     self.Agent.add_reward(0.0001)

        # upright reward
        # upright = self.hip_bone.up.dot(Vector3(0, 1, 0))
        # self.Agent.add_reward((upright) * 0.001)


        # # punish high vertical velocity
        # self.Agent.add_reward(-abs(self.hip_bone.Rigidbody.velocity.y) * 0.01)


        # elif self.hip_bone.Legs.finished + self.feets[0].Legs.finished + self.feets[1].Legs.finished == 2:
        #     self.Agent.add_reward(0.005)

    def addRewardByVelocity(self):
        direction = self.goal.position - self.hip_bone.position
        direction.y = 0

        distance = direction.magnitude()
        if distance < 0.001:
            return

        direction = direction.normalized()

        vel = Vector3(self.body_vel.x, 0, self.body_vel.z)
        speed_toward_goal = vel.dot(direction)

        # choose wanted speed by distance
        if distance >= 1.0:
            self.target_speed = 2
        else:
            self.target_speed = 0.5

        # reward being close to target speed
        speed_error = abs(speed_toward_goal - self.target_speed)
        speed_reward = 1.0 - np.clip(speed_error / self.target_speed, 0, 1)

        self.add_reward(speed_reward * 0.003)

        up_direction = self.goal.position.y - self.hip_bone.position.y
        up_velocity = self.hip_bone.Rigidbody.velocity.y

        reward = np.sign(up_direction) * up_velocity
        self.add_reward(np.clip(reward, -1, 1) * 0.005)




    def addRewardByDistance(self):
        dis = self.getDistance()
        progress = self.lastDistance - dis
        self.lastDistance = dis

        self.add_reward(progress * 0.005)

    def set_body_val(self):
        total_mass = 0.0
        vel = Vector3()

        for part in self.body_parts:
            mass = part.Rigidbody.mass
            vel += part.Rigidbody.velocity * mass
            total_mass += mass

        self.body_vel = vel / total_mass

    def move(self, action, dt):
        for i, servo in enumerate(self.servos):
            servo.ServoController.move(action[i] * self.speed, dt)

    def getDistance(self):
        hip_dis = (self.hip_bone.position - self.goal.children[0].position).magnitude()

        # foot1_vec = self.feets[0].position - self.goal.children[1].position
        # foot1_vec.y = 0
        # foot1_dis = foot1_vec.magnitude()

        # foot2_vec = self.feets[1].position - self.goal.children[2].position
        # foot2_vec.y = 0
        # foot2_dis = foot2_vec.magnitude()

        return hip_dis # + foot1_dis * 0.25 + foot2_dis * 0.25

    def add_observations(self):
        for body_part in self.body_parts:
            self.add_observation(body_part.local_position)
            self.add_observation(body_part.quaternion.normalized())
            self.add_observation(body_part.Rigidbody.velocity)
            self.add_observation(body_part.Rigidbody.angular_velocity)
        for servo in self.servos:
            self.add_observation(servo.ServoController.target_angle)


        self.add_observation(self.goal.children[0].local_position)

        self.add_observation(self.body_vel)
        self.add_observation(self.target_speed)
        self.add_observation(self.parent.findTheCenterOfMass())
        self.add_observation(self.feets[0].Collider.stay)
        self.add_observation(self.feets[1].Collider.stay)

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


class BodyPart(Component):
    def __init__(self, agent, other):
        super(BodyPart, self).__init__()
        self.Agent = agent
        self.other = other

    def OnCollisionEnter(self, Collision):
        if Collision.other.parent.get_component("Wall"):
            self.Agent.add_reward(-2)
            self.Agent.end_episode()
            self.other.fall = True


class Legs(Component):

    def __init__(self, agent, other):
        super(Legs, self).__init__()
        self.Agent = agent
        self.other = other
        self.dt = 0
        self.reward = 0
        self.finished = False

    def OnCollisionEnter(self, Collision):
        if (Collision.other.parent in self.other.goal.children) and not self.finished:
            self.Agent.add_reward(2)
            # self.other.fall = False
            self.other.next_goal()
            self.Agent.end_episode()


    def OnCollisionStay(self, Collision):
        if Collision.other.parent in self.other.goal.children:
            self.finished = True

    def OnCollisionExit(self, Collision):
        if (Collision.other.parent in self.other.goal.children) and self.finished:
            self.finished = False

    def Reset(self):
        self.finished = False