from bereshit.addons.PPO import Config, Trainer, Agent
from bereshit import Object, Vector3, Core, Camera, BoxCollider, Rigidbody
from bereshit.addons.essentials import FPS_cam, CamController
from Names_types import Wall, Goal
from MoveToGoal import MoveToGoal

cam = Object(position=Vector3(0, 5, 0), rotation=Vector3(90,0,0)).add_component(Camera(), CamController())


floor = Object(position=Vector3(0,-1,0), size=Vector3(10,1,10)).add_component(BoxCollider(), Rigidbody(isKinematic=True, friction_coefficient=1))
wall1 = Object(position=Vector3(-5,0,0), size=Vector3(1,1,10)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())
wall2 = Object(position=Vector3(5,0,0), size=Vector3(1,1,10)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())
wall3 = Object(position=Vector3(0,0,-5), size=Vector3(10,1,1)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())
wall4 = Object(position=Vector3(0,0,5), size=Vector3(10,1,1)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Wall())

goal = Object(position=Vector3(2,0,0)).add_component(BoxCollider(), Rigidbody(isKinematic=True), Goal())

config = Config(
    obs_dim=6,
    action_dim_continuous=2,
    rollout_steps=1024,
    device="cpu",
    best_model_path="data/model.pt",

)

trainer = Trainer(config)

agent_component = Agent(trainer, agent_id=0)
agent = Object().add_component(BoxCollider(), Rigidbody(Freeze_Rotation=Vector3(1,1,1)), agent_component, MoveToGoal(goal))

scene = [floor, wall1, wall2, wall3, wall4, agent, goal]

Core.run([cam] + scene, speed=10, Render=True)
