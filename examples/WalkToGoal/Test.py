import keyboard

from bereshit import Component


class BendDown(Component):
    def __init__(self, servo1, servo2, servo3, servo11, servo22, servo33):
        super(BendDown, self).__init__()
        self.speed = 200
        self.servo1 = servo1
        self.servo2 = servo2
        self.servo3 = servo3
        self.servo11 = servo11
        self.servo22 = servo22
        self.servo33 = servo33

    def Update(self, dt):
        if keyboard.is_pressed("k"):
            self.servo1.ServoController.move(-self.speed, dt)
            self.servo2.ServoController.move(self.speed * 2, dt)
            self.servo3.ServoController.move(-self.speed, dt)

        if keyboard.is_pressed("h"):
            self.servo11.ServoController.move(-self.speed, dt)
            self.servo22.ServoController.move(self.speed * 2, dt)
            self.servo33.ServoController.move(-self.speed, dt)

        if keyboard.is_pressed("i"):
            self.servo1.ServoController.move(self.speed, dt)
            self.servo2.ServoController.move(-self.speed * 2, dt)
            self.servo3.ServoController.move(self.speed, dt)

        if keyboard.is_pressed("y"):
            self.servo11.ServoController.move(self.speed, dt)
            self.servo22.ServoController.move(-self.speed * 2, dt)
            self.servo33.ServoController.move(self.speed, dt)

        if keyboard.is_pressed("u"):
            self.servo11.ServoController.move(self.speed, dt)




        if keyboard.is_pressed("q"):
            self.servo1.ServoController.move(-self.speed, dt)
            self.servo2.ServoController.move(self.speed*2, dt)
            self.servo3.ServoController.move(-self.speed, dt)

            self.servo11.ServoController.move(-self.speed, dt)
            self.servo22.ServoController.move(self.speed * 2, dt)
            self.servo33.ServoController.move(-self.speed, dt)

        if keyboard.is_pressed("e"):
            self.servo1.ServoController.move(self.speed, dt)
            self.servo2.ServoController.move(-self.speed*2, dt)
            self.servo3.ServoController.move(self.speed, dt)

            self.servo11.ServoController.move(self.speed, dt)
            self.servo22.ServoController.move(-self.speed * 2, dt)
            self.servo33.ServoController.move(self.speed, dt)
