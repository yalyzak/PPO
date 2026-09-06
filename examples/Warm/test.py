import keyboard
from bereshit import Component

class Test(Component):
    def __init__(self, servos):
        super(Test, self).__init__()
        self.servos = servos

    def Update(self, dt):
        if keyboard.is_pressed("h"):
            for servo in self.servos:
                servo.ServoController.move(10, dt)

        if keyboard.is_pressed("g"):
            for servo in self.servos:
                servo.ServoController.move(-10, dt)

from bereshit.addons.PPO import Config, Academy
from bereshit import GameObject, Vector3, Core, Camera, BoxCollider, Rigidbody, FixedJoint
from bereshit.addons.essentials import FPS_cam, CamController, Servo
from bereshit.addons.PPO.essentials import Goal, Wall
from MoveToGoal import MoveToGoal

cam = GameObject(position=Vector3(0, 5, 0), rotation=Vector3(90,0,0)).add_component(Camera(shading="material preview"), CamController(), FPS_cam())

floor = GameObject(position=Vector3(0,-1,0), size=Vector3(30,1,30)).add_component(BoxCollider(), Rigidbody(isKinematic=True))

wall1 = GameObject(position=Vector3(-15.5,0,0), size=Vector3(1,1,30)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())
wall2 = GameObject(position=Vector3(15.5,0,0), size=Vector3(1,1,30)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())
wall3 = GameObject(position=Vector3(0,0,-15.5), size=Vector3(30,1,1)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())
wall4 = GameObject(position=Vector3(0,0,15.5), size=Vector3(30,1,1)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())

max = 45
min = -45
speed = 10
torque = 5

goal = GameObject(position=Vector3(2,-0.4,2), size=Vector3(0.3, 0.3, 0.3)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Goal())

spine1 = GameObject(size=Vector3(0.5,0.5,0.5), name="spine1").add_component(BoxCollider(), Rigidbody(mass=0.05))

spine2 = GameObject(position=Vector3(1,0,0), size=Vector3(0.5,0.5,0.5), name="spine2").add_component(BoxCollider(), Rigidbody(mass=0.05), Servo(spine1, Vector3(0,0,1), max, min, speed, torque))

spine3 = GameObject(position=Vector3(2,0,0), size=Vector3(0.5,0.5,0.5), name="spine3").add_component(BoxCollider(), Rigidbody(mass=0.05), Servo(spine2, Vector3(0,1,0), max, min, speed, torque))

spine4 = GameObject(position=Vector3(3,0,0), size=Vector3(0.5,0.5,0.5), name="spine3").add_component(BoxCollider(), Rigidbody(mass=0.05), Servo(spine3, Vector3(0,0,1), max, min, speed, torque))

head = GameObject(position=Vector3(4,0,0), size=Vector3(0.5,0.5,0.5), name="head").add_component(BoxCollider(), Rigidbody(mass=0.05), FixedJoint(spine4))


head.add_component(Test([spine4, spine3, spine3, spine2]))

warm = GameObject(size=Vector3(), children=[spine1, spine2, spine3, spine4, head])

scene = GameObject(size=Vector3(), children=[floor, wall1, wall2, wall3, wall4, goal, warm, cam])#.add_component(ServoMovment(spine2))


Core.run([scene], Render=True)
