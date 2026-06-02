from bereshit import Object, Vector3, Core, Camera
from bereshit.addons.essentials import FPS_cam, CamController
from bereshit.addons.PPO.examples.WalkToGoal.creat_robot import creat_robot

cam = Object(position=Vector3(8, 0, 0), rotation=Vector3(0,-90,0)).add_component(Camera(shading="material preview"), CamController(), FPS_cam())

Core.run([creat_robot(model="walk.pt")] + [cam], Render=True, tick=1/240, speed=1)
