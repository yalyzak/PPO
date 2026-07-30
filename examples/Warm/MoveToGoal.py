import random
from bereshit import Vector3
from bereshit.addons.PPO import Agent


class MoveToGoal(Agent):
    def __init__(self, goal):
        super(MoveToGoal, self).__init__()
        self.goal = goal
        self.speed = 120
        self.last_distance = Vector3()

    def attach(self, parent):
        self.bodyParts = parent.get_all_children_physics()
        self.servos = parent.search_by_component("Servo")
        self.head = parent.search_by_name("head")[0]

    def get_distance(self):
        return (self.head.transform.local_position - self.goal.transform.local_position).magnitude()

    def OnEpisodeBegin(self):
        self.parent.reset_to_default()
        self.goal.reset_to_default()
        self.parent.transform.local_position += Vector3(random.uniform(-10, 10), 0, random.uniform(0, 10))
        self.goal.transform.local_position += Vector3(random.uniform(-10, 10), 0, random.uniform(-10, -5))
        self.last_distance = self.get_distance()

    def add_observations(self):
        for body_part in self.bodyParts:
            self.add_observation(body_part.transform.local_position)
            self.add_observation(body_part.Rigidbody.velocity)
            self.add_observation(body_part.Rigidbody.angular_velocity)

        if body_part in self.servos:
            self.add_observation(body_part.ServoController.get_target_angle())

        self.add_observation(self.goal.transform.local_position)

    def move(self, actions, dt):
        for i, servo in enumerate(self.servos):
            servo.ServoController.move(actions[i] * self.speed, dt)

    def addRewardByDistance(self):
        distance = self.get_distance()
        delta_distance = self.last_distance - distance
        self.last_distanc = distance
        reward = delta_distance * 0.01
        self.add_reward(reward)

    def Update(self, dt):
        self.add_observations()
        actions = self.get_continuous_actions()
        self.move(actions, dt)
        self.addRewardByDistance()
        self.add_reward(-0.001)


