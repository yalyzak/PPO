from bereshit import BoxCollider, HingeJoint, FixedJoint, MeshRander
from bereshit.addons.PPO.examples.WalkToGoal.ServoController import ServoController, ServoControllerKeyboard


class Servo:
    def __init__(self, mount, axis, use_model=False, useKeyboard=False, max_rotation=90, min_rotation=-90):
        self._servo = None
        self._mount = mount
        self._axis = axis
        self._max_r = max_rotation
        self._min_r = min_rotation
        self.use_model = use_model
        self.useKeyboard = useKeyboard

    def attach(self, parent):
        self._servo = parent

        if self.useKeyboard:
            self._servo.add_component(
                HingeJoint(self._mount, self._axis, max_rotation=self._max_r, min_rotation=self._min_r),
                ServoControllerKeyboard(self._mount, max=self._max_r, min=self._min_r))
        else:
            self._servo.add_component(
                HingeJoint(self._mount, self._axis, max_rotation=self._max_r, min_rotation=self._min_r),
                ServoController(self._mount, max_rotation=self._max_r, min_rotation=self._min_r))

        if self.use_model:
            self._servo.add_component(MeshRander(obj_path="data/servo.glb"))
        return "Servo"



