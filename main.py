import numpy
import tensorflow
import rl
from cell import Cell
import random
from matplotlib import pyplot as plt

# TODO:
# 1) costruzione env
#        1.1 - griglia
#        1.2 - mura
# 2) agents
#        2.1 - actions [up, down, right, left, sense]
#        2.2 - rewards
# 3) policy
#        3.1 - random --> inferenza sui risultati di ricerca
#        3.2 - complete search strategy (DFS) ---> confronto con strategia random
# 4) Analysis:
#        4.1 - hours to learn tasks
#        4.2 - repeated experiments
#        4.3 - differences of hiding strategies depending on searching strategy

cell = Cell(2, 3)
print(cell)

actions = [0,1,2,3] #, cell.sense()]
rnd = random.choice(actions)

print(cell.up())
print(cell.get_y())
plt.grid(20)
plt.plot(cell.get_x(), cell.get_y(), 'yo')

plt.show()



'''
def grid(self, num_rows, num_cols, num_walls):

    self.rows = num_rows
    self.cols = num_cols
    self.walls = num_walls
    return
'''
