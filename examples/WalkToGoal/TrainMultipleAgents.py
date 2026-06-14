from bereshit import Vector3, Core
from bereshit.addons.PPO.examples.WalkToGoal.creat_robot import creat_robot

legs = [creat_robot(Vector3(0,0,i * 200), model="walk.pt") for i in range(1)]


Core.run(legs, Render=False, tick=1/240, speed=1)
