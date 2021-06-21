class Cell():

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.wall = False
        self.explored = False

    def up(self):
        self.y = self.y + 1
        return self

    def down(self):
        self.y = self.y - 1
        return self

    def right(self):
        self.x = self.x+1
        return self

    def left(self):
        self.x = self.x-1
        return self

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y
