import keyboard
from bereshit import Component

class ServoMovment(Component):
    def __init__(self, servo):
        super(ServoMovment, self).__init__()
        self.servo = servo

    def Update(self, dt):
        if keyboard.is_pressed("q"):
            self.servo.ServoController.move(-50, dt)

        elif keyboard.is_pressed("e"):
            self.servo.ServoController.move(50, dt)

