from bereshit import BoxCollider, HingeJoint, FixedJoint, MeshRander
from ServoController import ServoController, ServoControllerKeyboard

class Servo:
    def __init__(self, mount, axis, use_model=False, useKeyboard=False):
        self._servo = None
        self._mount = mount
        self._axis = axis
        self._max_r = 90
        self._min_r = -90
        self.use_model = use_model
        self.useKeyboard = useKeyboard

    def attach(self, parent):
        self._servo = parent

        if self.useKeyboard:
            self._servo.add_component(
                HingeJoint(self._mount, self._axis),
                ServoControllerKeyboard(self._mount))
        else:
            self._servo.add_component(
                HingeJoint(self._mount, self._axis),
                ServoController(self._mount))

        if self.use_model:
            self._servo.add_component(MeshRander(obj_path="data/servo.glb"))
        return "Servo"



