import copy

from bereshit import Object, Vector3, Core, Camera, BoxCollider, Rigidbody, FixedJoint, Joint
from bereshit.addons.essentials import FPS_cam, CamController
from Servo import Servo
from WalkToGoal import Walk
from bereshit.addons.PPO.examples.MoveToGoal.Names_types import Goal, Wall
from bereshit.addons.PPO import Trainer, Agent, Config

cam = Object(position=Vector3(8, 0, 0), rotation=Vector3(0,-90,0)).add_component(Camera(shading="material preview"), CamController(), FPS_cam())


def creat_robot(pos):
    floor = Object(size=Vector3(100, 1, 100), position=Vector3(0, -1, 0)).add_component(BoxCollider(),
                                                                                        Rigidbody(isKinematic=True), Wall())

    goal = Object(position=Vector3(-1, 0, 4)).add_component(BoxCollider(is_trigger=True), Rigidbody(isKinematic=True),
                                                           Goal())
    scene = [floor, goal]

    feet = Object(size=Vector3(2.6, 0.5, 1.2), position=Vector3(-.5,0,-0.3), name="feet").add_component(BoxCollider(), Rigidbody(mass=0.01))

    mount = Object(position=Vector3(0,1,0), size=Vector3(.5, .5, .5), name="mount").add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(feet))

    servo1 = Object(position=Vector3(0,2,0), size=Vector3(.5, .5, .5), name="ankle1").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(mount, axis=Vector3(0,0,1)))

    servo2 = Object(position=Vector3(0,3,0), size=Vector3(.5, .5, .5), name="ankle2").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(servo1, axis=Vector3(1,0,0)))

    calf = Object(position=Vector3(0,5,0), size=Vector3(.5,3,.5)).add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(servo2))

    mount2 = Object(position=Vector3(0,7,0), size=Vector3(.5, .5, .5)).add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(calf))

    knee = Object(position=Vector3(0,8,0), size=Vector3(.5, .5, .5), name="knee").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(mount2, axis=Vector3(0,0,1)))

    thigh = Object(position=Vector3(0,10,0), size=Vector3(.5,3,.5)).add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(knee))

    hip1 = Object(position=Vector3(0,12,0), size=Vector3(.5, .5, .5), name="hip1").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(thigh, axis=Vector3(0,0,1)))

    hip2 = Object(position=Vector3(0,13,0), size=Vector3(.5, .5, .5), name="hip2").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(hip1, axis=Vector3(1,0,0)))


    leg = Object(size=Vector3(), children=[feet, mount, servo1, servo2, calf, mount2, knee, thigh, hip1, hip2])

    leg2 = copy.deepcopy(leg)

    feet2 = leg2.search_by_name("feet")[0]
    mount2 = leg2.search_by_name("mount")[0]
    feet2.position.z += 0.6

    mount2.Joint.cast_anchor()

    leg2.local_position += Vector3(0,0,2)

    leg2.set_default()

    hip_l = leg2.search_by_name("hip2")[0]


    hip_bone = Object(position=Vector3(0,14,1), size=Vector3(.5, .5, 2), name="hip_bone").add_component(BoxCollider(), Rigidbody(mass=0.02), FixedJoint(hip_l), FixedJoint(hip2))

    config = Config(
        obs_dim=302,
        action_dim_continuous=10,
        rollout_steps=1024,
        device="cpu",
        best_model_path="walk.pt",
        max_steps=1000,
        hidden_size=256

    )

    trainer = Trainer(config)

    agent_component = Agent(trainer, agent_id=0)

    legs = Object(size=Vector3(), children=[leg, leg2, hip_bone] + scene).add_component(agent_component, Walk(goal))

    legs.local_position += pos

    legs.set_default()

    return legs

legs = [creat_robot(Vector3(0,0,i * 200)) for i in range(1)]


Core.run(legs + [cam], Render=True, tick=1/240, speed=1)
