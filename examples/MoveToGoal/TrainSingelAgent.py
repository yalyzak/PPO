from bereshit.addons.PPO import Config, Trainer, Agent, Academy
from bereshit import GameObject, Vector3, Core, Camera, BoxCollider, Rigidbody
from bereshit.addons.essentials import FPS_cam, CamController
from bereshit.addons.PPO.essentials import Goal, Wall
from MoveToGoal import MoveToGoal

cam = GameObject(position=Vector3(0, 5, 0), rotation=Vector3(90,0,0)).add_component(Camera(), CamController(), FPS_cam())

floor = GameObject(position=Vector3(0,-1,0), size=Vector3(50,1,50)).add_component(BoxCollider(), Rigidbody(isKinematic=True, friction_coefficient=1))
wall1 = GameObject(position=Vector3(-25,0,0), size=Vector3(1,1,50)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())
wall2 = GameObject(position=Vector3(25,0,0), size=Vector3(1,1,50)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())
wall3 = GameObject(position=Vector3(0,0,-25), size=Vector3(50,1,1)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())
wall4 = GameObject(position=Vector3(0,0,25), size=Vector3(50,1,1)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())

goal = GameObject(position=Vector3(2,0,0)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Goal())

config = Config(
    obs_dim=9,
    action_dim_continuous=2,
    rollout_steps=5000,
    device="cpu",
    best_model_path="model.pt",
    max_steps=50

)


Academy.setup_trainer(config)

agent = GameObject().add_component(BoxCollider(), Rigidbody(Freeze_Rotation=Vector3(1,1,1)), MoveToGoal(goal))

main = GameObject(children=[floor, wall1, wall2, wall3, wall4, goal, agent], size=Vector3(0,0,0))

scene = [main, cam]

Core.run_max_speed(scene, Render=False, MaxTime=60*60)  # train for 1 simulated hour
