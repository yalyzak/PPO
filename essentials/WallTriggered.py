from bereshit import Component

class WallTriggered(Component):
    def OnCollisionEnter(self, Collision):
        if Collision.other.parent.get_component("Wall"):
            self.parent.parent.MoveToGoal.add_reward(-1)
            self.parent.parent.MoveToGoal.end_episode()
