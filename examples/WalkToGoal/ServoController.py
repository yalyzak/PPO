import math

import keyboard
from bereshit import Vector3, Quaternion, HingeJoint


class ServoController:
    def __init__(self, other, max=90, min=-90):
        self.target_angle = 0.0
        self.kP = 0.2  # converts angle error → desired speed
        self.kD = 0.1  # constant speed
        self.max_speed = 60.54  # rad/s
        self._speed = 120
        self.max_torque = 100.32  # strength of motor
        self._axis = Vector3()
        self.max_rotation = max
        self.min_rotation = min
        self.other = other

    def Reset(self):
        self.target_angle = 0.0

    def move(self, input_value, dt):
        """
        input_value: float in range [-1, 1]
            -1 → full negative (e.g. "w")
             0 → no input
             1 → full positive (e.g. "s")
        """

        self._axis = self.parent.get_component(HingeJoint).axis_world.normalized()

        # smooth continuous control
        self.target_angle += input_value * self._speed * dt

        # clamp rotation
        self.target_angle = max(min(self.target_angle, self.max_rotation), self.min_rotation)

        self.fix(dt)

    def fix(self, dt):
        self._axis = self.parent.get_component(HingeJoint).axis_world.normalized()

        rb = self.parent.Rigidbody

        relative_q = self.other.quaternion.conjugate() * self.parent.quaternion
        current_angle = relative_q.to_euler().dot(self._axis)
        error = self.target_angle - current_angle

        # normalize angle to [-180, 180] (VERY IMPORTANT)
        error = (error + 180) % 360 - 180

        angular_velocity = rb.angular_velocity.dot(self._axis)

        # torque (this creates acceleration naturally)
        torque = self.kP * error - self.kD * angular_velocity

        # clamp torque
        torque = max(min(torque, self.max_torque), -self.max_torque)

        impulse = torque * dt * self._axis
        self.apply_angular_impulse(impulse)
        # self.clamp_speed()

    def clamp_rotation(self):
        self.parent.quaternion = max(min(self.parent.quaternion.to_euler(), self.max_rotation), -self.max_rotation)

    def clamp_speed(self):
        self.parent.Rigidbody.angular_velocity.z = max(min(self.parent.Rigidbody.angular_velocity.z, self.max_speed),
                                                       -self.max_speed)

    def apply_angular_impulse(self, impulse: Vector3):
        rb = self.parent.Rigidbody
        delta_w = rb.Iinv_world() @ impulse.to_np()
        rb.angular_velocity += Vector3.from_np(delta_w)

    def PhysicsUpdate(self, dt):
        self.fix(dt)

class ServoControllerKeyboard(ServoController):
    def __init__(self, other, keys=["e", "q"]):
        super().__init__()
        self.key0 = keys[0]
        self.key1 = keys[1]

    @staticmethod
    def get_keyboard_input(key_neg, key_pos):
        value = 0.0
        if keyboard.is_pressed(key_neg):
            value -= 1.0
        if keyboard.is_pressed(key_pos):
            value += 1.0
        return value

    def PhysicsUpdate(self, dt):
        super().Update(dt)
        input_value = ServoControllerKeyboard.get_keyboard_input(self.key0, self.key1)
        if input_value != 0:
            self.move(input_value, dt)

