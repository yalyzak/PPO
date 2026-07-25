from bereshit.addons.PPO import Agent
from bereshit import Vector3
import random


class MoveToGoal(Agent):
    def __init__(self, goal):
        super(MoveToGoal, self).__init__()
        self.goal = goal
        self.speed = 500
        self.dt = 0

    def OnEpisodeBegin(self):
        self.parent.reset_to_default()
        self.goal.reset_to_default()
        self.parent.transform.local_position += Vector3(random.uniform(-3, 3), 0, random.uniform(-3, 3))
        self.goal.transform.local_position += Vector3(random.uniform(-3, 3), 0, random.uniform(-3, 3))

    def Update(self, dt):
        self.dt += dt
        pos = self.parent.transform.local_position
        pos2 = self.goal.transform.local_position
        self.add_observation(pos)
        self.add_observation(pos2)
        action = self.get_continuous_actions()
        self.Move(action[0], action[1], dt)
        self.addRewardByDistance(pos, pos2)
        if self.dt > 20:
            stats = self.trainer.learn_if_ready()
            if stats is not None:
                print("PPO update:", stats)
                self.trainer.print_performance()
                self.dt = 0

    def Move(self, x, z, dt):
        self.parent.Rigidbody.velocity += Vector3(x, 0, z) * dt * self.speed

    def addRewardByDistance(self, pos, pos2):
        distance = (pos - pos2).magnitude()
        reward = -distance * 0.01
        self.add_reward(reward)

    def OnCollisionEnter(self, Collision):
        if Collision.other.parent.get_component("Wall"):
            self.add_reward(-1)
            self.end_episode()
        elif Collision.other.parent.get_component("Goal"):
            self.add_reward(1)
            self.end_episode()
