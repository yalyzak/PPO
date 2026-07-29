import random

from bereshit import Component, Vector3

class MoveToGoal(Component):
    def __init__(self, goal):
        super(MoveToGoal, self).__init__()
        self.goal = goal
        self.speed = 1
        self.last_distance = Vector3()

    def attach(self, parent):
        self.servos = parent.search_by_component("Servo")
        self.haed = parent.search_by_name("head")

    def get_distance(self):
        return (self.haed - self.goal).magnitude()

    def OnEpisodeBegin(self):
        self.parent.reset_to_default()
        self.goal.reset_to_default()
        self.parent.transform.local_position += Vector3(random.uniform(-3, 3), 0, random.uniform(-3, 3))
        self.goal.transform.local_position += Vector3(random.uniform(-3, 3), 0, random.uniform(-3, 3))
        self.last_distance = get_distance()
    def add_observations(self):
        for body_part in self.servos:
            self.add_observation(body_part.transform.local_position)
            self.add_observation(body_part.transform.rotation)
            self.add_observation(body_part.Rigidbody.velocity)
            self.add_observation(body_part.Rigidbody.angular_velocity)

        self.add_observation(self.goal.transform.local_position)

    def move(self, actions, dt):
        for i, servo in enumerate(self.servos):
            servo.ServoController.move(actions[i] * self.speed, dt)
    def addRewardByDistance(self):
        distance = (pos - pos2).magnitude()
        reward = -distance * 0.01
        self.add_reward(reward)

    def Update(self, dt):
        self.add_observations()
        actions = self.get_continuous_actions()
        self.Move(actions, dt)
        self.addRewardByDistance()



