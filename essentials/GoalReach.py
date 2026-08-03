from bereshit import Component

class GoalReach(Component):
    def __init__(self, agent):
        super(GoalReach, self).__init__()
        self.agent = agent

    def OnCollisionEnter(self, Collision):
        if Collision.other.parent.get_component("Wall"):
            self.agent.add_reward(-1)
            self.agent.end_episode()
        elif Collision.other.parent.get_component("Goal"):
            self.agent.add_reward(1)
            self.agent.end_episode()