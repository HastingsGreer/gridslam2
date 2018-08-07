
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (15, 15)

import scipy.optimize




#https://www.opengl.org/discussion_boards/showthread.php/197893-View-and-Perspective-matrices

import math
import numpy as np
 
def magnitude(v):
    return math.sqrt(np.sum(v ** 2))
 
def normalize(v):
    m = magnitude(v)
    if m == 0:
        return v
    return v / m
 


def simplePerspective(a):
    return np.matrix([[1,0,0,0],
                      [0,1,0,0],
                      [0,0,1,a],
                      [0,0,-a,0]])

 
def translate(xyz):
    x, y, z = xyz
    return np.matrix([[1,0,0,x],
                      [0,1,0,y],
                      [0,0,1,z],
                      [0,0,0,1]])

def sincos(a):
    a = math.radians(a)
    return math.sin(a), math.cos(a)
 
def rotate(a, xyz):
    x, y, z = normalize(xyz)
    s, c = sincos(a)
    nc = 1 - c
    return np.matrix([[x*x*nc +   c, x*y*nc - z*s, x*z*nc + y*s, 0],
                      [y*x*nc + z*s, y*y*nc +   c, y*z*nc - x*s, 0],
                      [x*z*nc - y*s, y*z*nc + x*s, z*z*nc +   c, 0],
                      [           0,            0,            0, 1]])
 
def rotx(a):
    s, c = sincos(a)
    return np.matrix([[1,0,0,0],
                      [0,c,-s,0],
                      [0,s,c,0],
                      [0,0,0,1]])
 
def roty(a):
    s, c = sincos(a)
    return np.matrix([[c,0,s,0],
                      [0,1,0,0],
                      [-s,0,c,0],
                      [0,0,0,1]])
 
def rotz(a):
    s, c = sincos(a)
    return np.matrix([[c,-s,0,0],
                      [s,c,0,0],
                      [0,0,1,0],
                      [0,0,0,1]])
def euler(rotation):
    return rotx(rotation[0])*roty(rotation[1]) * rotz(rotation[2])
def matrix(translation, rotation, focalLength):
    return simplePerspective(focalLength) * euler(rotation) * translate(translation)

def projectPoints(points, matrix, image):
    out = matrix * points
    out /= out[ 3:4].copy()
    out = out[:2]
    out[0] += image.shape[1] / 2
    out[1] += image.shape[0] / 2
    return np.array(out, dtype=np.float)

def mat_from_vec(vector):
    translation = vector[0:3]
    rotation = vector[3:6]
    focal = 1/vector[6]
    the_matrix = matrix(translation, rotation, focal)
    return the_matrix
    
class ManualAligner:
    def __init__(self, image):
        self.image = image
        self.vector = np.array([0, 0, -15, 0, 0, 0, 800])
        d = 3
        gridx, gridy = np.mgrid[-d:d:1, -d:d:1]
        self.grid = np.array([gridx.flatten(), gridy.flatten(), np.zeros(((2*d)**2)),np.ones(((2*d)**2)) ])


        mymatrix = mat_from_vec(self.vector)

        self.pts = projectPoints(self.grid, mymatrix, self.image)

        self.gridOnScreen = np.zeros(self.pts.shape)
        
        self.fig = plt.figure()

        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.image)
        self.ax.set_title('click on points')

        self.fittedGrid, = self.ax.plot(self.pts[0], self.pts[1], "o", picker=5)  # 5 points tolerance

        self.trueGrid, = self.ax.plot(self.gridOnScreen[0], self.gridOnScreen[1], "o")


        self.activeIndex = -1

        self.is_pick = False
        self.fig.canvas.mpl_connect('pick_event', self.onpick)
        
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
       
        plt.show()


    def error(self, vector):
        the_matrix = mat_from_vec(vector)
        the_pts = projectPoints(self.grid, the_matrix, self.image)
        mask = self.gridOnScreen != 0
        rotation = vector[3:6]
        focal = 1/vector[6]
        error = np.sum((self.gridOnScreen[mask] - the_pts[mask])**2) + np.sum(np.abs(rotation)) / 200 + np.abs(focal - 800) / 10
        
        return error


    def get_best_vector(self):
        res = scipy.optimize.minimize(self.error, np.array([0, 0, -5, 0, 0, 0, 800]))
        
        return res.x

    def onpick(self, event):    
        ind = event.ind
        if self.activeIndex != ind:
            self.is_pick = True
    
        self.activeIndex = ind[0]
        

    def on_click(self, event):
       
        if self.is_pick:
            self.is_pick = False
            return
        self.gridOnScreen[:, self.activeIndex] = event.xdata, event.ydata
        self.trueGrid.set_data(self.gridOnScreen[0], self.gridOnScreen[1])
        self.fig.canvas.draw()
        
        self.vector = self.get_best_vector()
        mymatrix = mat_from_vec(self.vector)
        
        self.pts = projectPoints(self.grid, mymatrix, self.image)
        
        self.fittedGrid.set_data(self.pts[0], self.pts[1])
        self.fig.canvas.draw()

if __name__ == "__main__":
    t1 = np.array(Image.open("2unitsfromceiling.jpg"))[::4, ::4]
    result = ManualAligner(t1).vector
    print(result)
    t1 = np.array(Image.open("test.jpg"))[::4, ::4]  
    result = ManualAligner(t1).vector
    print(result)
    t1 = np.array(Image.open("floor.jpg"))[::4, ::4]  
    result = ManualAligner(t1).vector
    print(result)
        






