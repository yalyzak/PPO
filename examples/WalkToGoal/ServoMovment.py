import keyboard


class ServoMovment:
    def attach(self, parent):
        self.feets = parent.search_by_name("feet")
        self.ankle1s = parent.search_by_name("ankle1")
        self.ankle2s = parent.search_by_name("ankle2")
        self.knees = parent.search_by_name("knee")
        self.hip1s = parent.search_by_name("hip1")
        self.hip2s = parent.search_by_name("hip2")
        self.degres = 0

    def leanRight(self, dt):
        self.degres -= 1 * dt
        self.ankle2s[0].ServoController.move(self.degres, dt)
        self.ankle2s[1].ServoController.move(self.degres, dt)
        self.hip2s[0].ServoController.move(-self.degres, dt)
        self.hip2s[1].ServoController.move(-self.degres, dt)

    def leanLeft(self, dt):
        self.degres += 1 * dt
        self.ankle2s[0].ServoController.move(self.degres, dt)
        self.ankle2s[1].ServoController.move(self.degres, dt)
        self.hip2s[0].ServoController.move(-self.degres, dt)
        self.hip2s[1].ServoController.move(-self.degres, dt)

    def walkRight(self, dt):
        self.degres -= 1 * dt
        self.knees[0].ServoController.move(-2 * self.degres, dt)
        self.ankle1s[0].ServoController.move(self.degres, dt)
        # self.hip2s[0].ServoController.move(self.degres * 0.1, dt)

    def walkRight2(self, dt):
        self.degres += 1 * dt
        self.knees[0].ServoController.move(-2 * self.degres, dt)
        self.ankle1s[0].ServoController.move(self.degres, dt)

    def stepRight(self, dt):
        self.degres -= 1 * dt
        self.hip1s[0].ServoController.move(2 * self.degres, dt)

    def Update(self, dt):
        print(self.parent.findTheCenterOfMass() - self.feets[1].position)

        if keyboard.is_pressed("e"):
            self.leanRight(dt)

        if keyboard.is_pressed("q"):
            self.leanLeft(dt)

        if keyboard.is_pressed("r"):
            self.walkRight(dt)

        if keyboard.is_pressed("t"):
            self.walkRight2(dt)