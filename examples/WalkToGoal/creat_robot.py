import copy
from bereshit.addons.PPO.essentials import Goal, Wall
from bereshit import GameObject, Vector3, BoxCollider, Rigidbody, FixedJoint
from Servo import Servo
from WalkToGoal import Walk
from bereshit.addons.PPO import Config, Academy
from ServoMovment import ServoMovment

def creat_robot(pos=Vector3(), use_PPO=True, model=None, save=True, load_optimizer=True):
    floor = GameObject(size=Vector3(100, 1, 100), position=Vector3(0, -0.70, 0)).add_component(BoxCollider(),
                                                                                        Rigidbody(isKinematic=True), Wall())

    goal1 = GameObject(name="goal", position=Vector3(0, 14, 1), size=Vector3(0.5, 0.5, 0.5)).add_component(BoxCollider(is_trigger=True), Rigidbody(isKinematic=True),
                                                           Goal())
    # goal2 = Object(name="goal", position=Vector3(-1.5, 0, 2), size=Vector3(0.5, 0.5, 0.5)).add_component(BoxCollider(is_trigger=True),
    #                                                                                      Rigidbody(isKinematic=True),
    #                                                                                      Goal())
    # goal3 = Object(name="goal", position=Vector3(-1.5, 0, -0.5), size=Vector3(0.5, 0.5, 0.5)).add_component(BoxCollider(is_trigger=True),
    #                                                                                      Rigidbody(isKinematic=True),
    #                                                                                      Goal())
    goal = GameObject(size=Vector3(), position=Vector3(0,14,1), children=[goal1])

    scene = [floor, goal]

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

    leg2 = copy.deepcopy(leg)

    feet2 = leg2.search_by_name("feet")[0]
    mount2 = leg2.search_by_name("mount")[0]
    feet2.transform.position.z += 0.6

    mount2.Joint.cast_anchor()

    leg2.transform.local_position += Vector3(0,0,2)

    leg2.transform.set_default()

    hip_l = leg2.search_by_name("hip2")[0]


    hip_bone = GameObject(position=Vector3(0,14,1), size=Vector3(.5, .5, 2), name="hip_bone").add_component(BoxCollider(), Rigidbody(mass=0.02), FixedJoint(hip_l), FixedJoint(hip2))

    config = Config(
        obs_dim=295,
        action_dim_continuous=10,
        rollout_steps = 1024,
        device="cuda",
        hidden_size=256,
        max_steps=1000,
        best_model_path="walk.pt" if save else None,
        entropy_coef = 0.001,
        max_episode_reward = 30,
        # learning_rate = 5e-5,


    )

    Academy.setup_trainer(config)

    legs = GameObject(size=Vector3(), children=[leg, leg2, hip_bone] + scene)
    if use_PPO:
        legs.add_component(Walk(goal, scene))
    else:
        legs.add_component(ServoMovment())


    legs.transform.local_position += pos

    legs.set_default()

    return legs
