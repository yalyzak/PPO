import random

from bereshit import Vector3
from bereshit.addons.PPO import Agent
from bereshit.addons.PPO.essentials import GoalTriggered, WallTriggered


class MoveToGoal(Agent):
    def __init__(self, goal):
        super(MoveToGoal, self).__init__()
        self.success = 0
        self.episodes = 0
        self.speed = 120
        self.goal = goal

    def attach(self, parent):
        self.bodyParts = parent.get_all_children_physics()
        for bodypart in self.bodyParts:
            if bodypart.name != "feet":
                bodypart.add_component(WallTriggered(self))
            else:
                bodypart.add_component(GoalTriggered(self))

        self.servos = parent.search_by_component("Servo")
        self.head = parent.search_by_name("hip2")[0]
        self.feet = parent.search_by_name("feet")[0]

    def get_distance(self):
        return (self.feet.transform.local_position - self.goal.transform.local_position).magnitude()

    def OnEpisodeBegin(self):
        self.parent.reset_to_default()
        self.goal.reset_to_default()
        self.parent.transform.local_position += Vector3(random.uniform(-10, 10), 0, random.uniform(-10, 10))
        self.goal.transform.local_position += Vector3(random.uniform(-10, 10), 0, random.uniform(-10, 10))
        self.start_dis = self.get_distance()
        self.last_distance = self.start_dis
        self.episodes += 1

    def add_observations(self):
        for body_part in self.bodyParts:
            self.add_observation(body_part.transform.position)
            self.add_observation(body_part.Rigidbody.velocity)
            self.add_observation(body_part.Rigidbody.angular_velocity)
            self.add_observation(body_part.transform.quaternion)

            if body_part in self.servos:
                self.add_observation(body_part.ServoController.get_target_angle())

        self.add_observation(self.goal.transform.position)
        direction = (
                self.goal.transform.position
                - self.head.transform.position
        )

        self.add_observation(direction.normalized())
        self.add_observation(direction.magnitude() / 30.0)

    def move(self, actions, dt):
        for i, servo in enumerate(self.servos):
            servo.ServoController.move(actions[i] * self.speed, dt)

    def addRewardByDistance(self):
        distance = self.get_distance()
        reward = (self.last_distance - distance) / self.start_dis
        self.add_reward(reward)
        self.last_distance = distance

    def Update(self, dt):
        self.add_observations()
        actions = self.get_continuous_actions()
        self.endEarly()
        self.move(actions, dt)
        self.addRewardByDistance()
        if self.trainer.learn_if_ready():
            if self.episodes != 0:
                print(self.success/self.episodes)
            self.episodes = 0
            self.success = 0

    def endEarly(self):
        if self.feet.transform.position.y > 5 or self.feet.transform.position.y < -5:
            self.add_reward(-1)
            self.end_episode()