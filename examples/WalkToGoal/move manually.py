from bereshit import Object, Vector3, Core, Camera
from bereshit.addons.PPO.examples.WalkToGoal.creat_robot import creat_robot
from bereshit.addons.essentials import CamController, FPS_cam

cam = Object(position=Vector3(8, 0, 0), rotation=Vector3(0,-90,0)).add_component(Camera(shading="material preview"), CamController(), FPS_cam())

legs = [creat_robot(Vector3(0,0,i * 200), use_PPO=False) for i in range(1)]


Core.run(legs + [cam], Render=True, tick=1/240, speed=1, physics_epochs=20)
