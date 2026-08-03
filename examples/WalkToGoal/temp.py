import copy
from bereshit.addons.PPO.essentials import Goal, Wall, ServoMovment
from bereshit import GameObject, Vector3, BoxCollider, Rigidbody, FixedJoint, Core, Camera
from bereshit.addons.PPO.examples.WalkToGoal.MoveToGoal import Walk
from bereshit.addons.essentials import CamController, FPS_cam, Servo
from bereshit.addons.PPO import Config, Academy

def make():
    feet = GameObject(size=Vector3(2.6, 0.5, 1.2), position=Vector3(-.5,0,-0.3), name="feet").add_component(BoxCollider(), Rigidbody(mass=0.01))

    mount = GameObject(position=Vector3(0,1,0), size=Vector3(.5, .5, .5), name="mount").add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(feet))

    servo1 = GameObject(position=Vector3(0,2,0), size=Vector3(.5, .5, .5), name="ankle1").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(mount, axis=Vector3(0,0,1)))

    servo2 = GameObject(position=Vector3(0,3,0), size=Vector3(.5, .5, .5), name="ankle2").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(servo1, axis=Vector3(1,0,0)))

    calf = GameObject(position=Vector3(0,5,0), size=Vector3(.5,3,.5)).add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(servo2))

    mount2 = GameObject(position=Vector3(0,7,0), size=Vector3(.5, .5, .5)).add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(calf))

    knee = GameObject(position=Vector3(0,8,0), size=Vector3(.5, .5, .5), name="knee").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(mount2, axis=Vector3(0,0,1)))

    thigh = GameObject(position=Vector3(0,10,0), size=Vector3(.5,3,.5)).add_component(BoxCollider(), Rigidbody(mass=0.01), FixedJoint(knee))

    hip1 = GameObject(position=Vector3(0,12,0), size=Vector3(.5, .5, .5), name="hip1").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(thigh, axis=Vector3(0,0,1)))

    hip2 = GameObject(position=Vector3(0,13,0), size=Vector3(.5, .5, .5), name="hip2").add_component(BoxCollider(), Rigidbody(mass=0.01), Servo(hip1, axis=Vector3(1,0,0)))


    leg = GameObject(size=Vector3(), children=[feet, mount, servo1, servo2, calf, mount2, knee, thigh, hip1, hip2])

    return leg

leg = make()

leg2 = make()

feet2 = leg2.search_by_name("feet")[0]
mount2 = leg2.search_by_name("mount")[0]
feet2.transform.position.z += 0.6
mount2.FixedJoint.cast_anchor()
leg2.transform.local_position += Vector3(0, 0, 2)
leg2.transform.set_default()
hip_l = leg2.search_by_name("hip2")[0]
hip_r = leg.search_by_name("hip2")[0]

hip_bone = GameObject(position=Vector3(0,14,1), size=Vector3(.5, .5, 2), name="hip_bone").add_component(BoxCollider(), Rigidbody(mass=0.02), FixedJoint(hip_l), FixedJoint(hip_r))

goal = GameObject(name="goal", position=Vector3(0, 14, 1), size=Vector3(0.5, 0.5, 0.5)).add_component(BoxCollider(is_trigger=True), Rigidbody(isKinematic=True),Goal())

legs = GameObject(size=Vector3(), children=[leg, leg2, hip_bone, goal])

legs.set_default()

floor = GameObject(size=Vector3(100, 1, 100), position=Vector3(0, -0.70, 0)).add_component(BoxCollider(),
                                                                                    Rigidbody(isKinematic=True), Wall())

cam = GameObject(position=Vector3(8, 0, 0), rotation=Vector3(0,-90,0)).add_component(Camera(shading="material preview"), CamController(), FPS_cam())

config = Config(
        obs_dim=301,
        action_dim_continuous=10,
        rollout_steps = 16000,
        device="cuda",
        hidden_size=256,
        max_steps=500,
        best_model_path="walk.pt",
        entropy_coef = 0.001,
        max_episode_reward = 30,
        # learning_rate = 5e-5,


    )

Academy.setup_trainer(config)

legs.add_component(Walk(goal, hip_bone))
# Academy.load_model("walk.pt")

# legs.add_component(ServoMovment(leg.search_by_name("ankle2")[0]))

Core.run_max_speed([legs, floor], Render=False, tick=1/300)

