from bereshit import Component

class WallTriggered(Component):
    def __init__(self, agent):
        super(WallTriggered, self).__init__()
        self.agent = agent

    def OnCollisionEnter(self, Collision):
        if Collision.other.parent.get_component("Wall"):
            self.agent.add_reward(-1)
            self.agent.end_episode()
