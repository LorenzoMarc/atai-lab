ReadME:

requirements: 
The following packages are required to running the scripts:

# NN_resolver:
 - gym
 - numpy
 - tensorflow
 - time
 - matplotlib
 - keras-rl2

To run the training open a terminal on main.py location and use 
"python main.py"

# trees mazes resolution
 - typing
 - queu
 - PIL
 - opencv
 - functools
 - os

To generate and solve the maze use:


#dijkstra
    - queue                 #pip install queuelib
    - PIL                   #pip install Pillow
    - numpy                 #pip install numpy
    - random                #pip install random2
    - cv2                   #pip install opencv-python
    - typing                #pip install typing
    - time                  #pip install times
    - os                    #pip install os-sys
    - functools             #pip install functools

To generate and solve the maze with dijkstra execute the following command:

- run maze_generator.py (you can change settings in main)
- run dijkstra.py

The results can be seen graphically in the folders that have been generated.


# Simple Q-learn approach
Requirments for the Simple q-learning approach: 
 - gym
 - numpy

To generate the and solve the problem for a 5x5 maze, 
open the directory "QLearn" and execute "python main.py". This will train the Q-tables
and do the tests. A csv file is created with the data outputted from the programm.
This csv file is used as input for the "experiments.py" script that analyze the data.




