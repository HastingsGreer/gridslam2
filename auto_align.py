
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



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
    return  roty(rotation[1])* rotx(rotation[0]) * rotz(rotation[2])
def matrix(translation, rotation, focalLength):
    return simplePerspective(focalLength) * euler(rotation) * translate(translation)

def projectPoints(points, matrix, image):
    out = matrix * points
    out /= out[3:4].copy()
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

def error_point_compiled(world_point, screen_point, vector, image):

    x, y, z, pitch, roll, yaw, focallength = vector
    screen_x, screen_y = screen_point
    world_x, world_y, world_z, _ = world_point
    sin = np.sin
    cos = np.cos
    pi = np.pi
    width, height = image.shape
    print(width, height)
    return (((2*focallength*(world_x*(sin(pi*pitch/180)*sin(pi*roll/180)*sin(pi*yaw/180) 
        + cos(pi*roll/180)*cos(pi*yaw/180)) + world_y*(sin(pi*pitch/180)*sin(pi*roll/180)*cos(pi*yaw/180) 
        - sin(pi*yaw/180)*cos(pi*roll/180)) + world_z*sin(pi*roll/180)*cos(pi*pitch/180) 
        + x*(sin(pi*pitch/180)*sin(pi*roll/180)*sin(pi*yaw/180) + cos(pi*roll/180)*cos(pi*yaw/180)) 
        + y*(sin(pi*pitch/180)*sin(pi*roll/180)*cos(pi*yaw/180) - sin(pi*yaw/180)*cos(pi*roll/180)) 
        + z*sin(pi*roll/180)*cos(pi*pitch/180)) 
    + (-height + 2*screen_x)*(world_x*(sin(pi*pitch/180)*sin(pi*yaw/180)*cos(pi*roll/180) 
        - sin(pi*roll/180)*cos(pi*yaw/180)) + world_y*(sin(pi*pitch/180)*cos(pi*roll/180)*cos(pi*yaw/180) 
        + sin(pi*roll/180)*sin(pi*yaw/180)) + world_z*cos(pi*pitch/180)*cos(pi*roll/180) 
        + x*(sin(pi*pitch/180)*sin(pi*yaw/180)*cos(pi*roll/180) - sin(pi*roll/180)*cos(pi*yaw/180)) 
        + y*(sin(pi*pitch/180)*cos(pi*roll/180)*cos(pi*yaw/180) + sin(pi*roll/180)*sin(pi*yaw/180)) 
        + z*cos(pi*pitch/180)*cos(pi*roll/180)))**2 + (2*focallength*(world_x*sin(pi*yaw/180)*cos(pi*pitch/180) 
        + world_y*cos(pi*pitch/180)*cos(pi*yaw/180) - world_z*sin(pi*pitch/180) + x*sin(pi*yaw/180)*cos(pi*pitch/180) 
        + y*cos(pi*pitch/180)*cos(pi*yaw/180) - z*sin(pi*pitch/180)) 
        + (2*screen_y - width)*(world_x*(sin(pi*pitch/180)*sin(pi*yaw/180)*cos(pi*roll/180) 
            - sin(pi*roll/180)*cos(pi*yaw/180)) + world_y*(sin(pi*pitch/180)*cos(pi*roll/180)*cos(pi*yaw/180) 
            + sin(pi*roll/180)*sin(pi*yaw/180)) + world_z*cos(pi*pitch/180)*cos(pi*roll/180) 
            + x*(sin(pi*pitch/180)*sin(pi*yaw/180)*cos(pi*roll/180) - sin(pi*roll/180)*cos(pi*yaw/180)) 
            + y*(sin(pi*pitch/180)*cos(pi*roll/180)*cos(pi*yaw/180) + sin(pi*roll/180)*sin(pi*yaw/180)) 
            + z*cos(pi*pitch/180)*cos(pi*roll/180)))**2)/(4*(world_x*(sin(pi*pitch/180)*sin(pi*yaw/180)*cos(pi*roll/180) 
            - sin(pi*roll/180)*cos(pi*yaw/180)) + world_y*(sin(pi*pitch/180)*cos(pi*roll/180)*cos(pi*yaw/180) 
            + sin(pi*roll/180)*sin(pi*yaw/180)) + world_z*cos(pi*pitch/180)*cos(pi*roll/180) 
            + x*(sin(pi*pitch/180)*sin(pi*yaw/180)*cos(pi*roll/180) - sin(pi*roll/180)*cos(pi*yaw/180)) 
            + y*(sin(pi*pitch/180)*cos(pi*roll/180)*cos(pi*yaw/180) + sin(pi*roll/180)*sin(pi*yaw/180)) 
            + z*cos(pi*pitch/180)*cos(pi*roll/180))**2))

    
class ManualAligner:
    def __init__(self, image, focalLength=None, activeParams = np.array([True, True, True, True, True, True, True])):
        self.activeParams = activeParams
        self.focalLength = focalLength
        self.image = image
        self.vector = np.array([0, 0, -15, 0, 0, 0, 800])
        self.d = d = 6
        gridx, gridy = np.mgrid[-d:d:1, -d:d:1]
        self.grid = np.array([gridx.flatten(), gridy.flatten(), np.zeros(((2*d)**2)),np.ones(((2*d)**2)) ])
        mymatrix = mat_from_vec(self.vector)

        self.pts = projectPoints(self.grid, mymatrix, self.image)
        
        
        self.gridOnScreen = np.zeros(self.pts.shape)
    def interactive_align(self):
        self.fig = plt.figure()

        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.image)
        self.init_grid_display()

        self.ax.set_title('click on points')

        
        self.trueGrid, = self.ax.plot(self.gridOnScreen[0], self.gridOnScreen[1], "o")


        self.activeIndex = -1

        self.is_pick = False
        self.fig.canvas.mpl_connect('pick_event', self.onpick)
        
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
       
        plt.show()
    def align_from_saved(self, points_corresponding):
        self.gridOnScreen = points_corresponding
        self.vector[self.activeParams] = self.get_best_vector_active_params()
    def present_registration(self):
        self.fig = plt.figure()

        self.ax = self.fig.add_subplot(111)
        
        self.init_grid_display()
        
        self.ax.set_title('click on points')

        
        self.trueGrid, = self.ax.plot(self.gridOnScreen[0], self.gridOnScreen[1], "o")


        self.activeIndex = -1

        self.is_pick = False
        self.process_updated_annotation()
        self.ax.imshow(self.image)
        plt.show()
    def init_grid_display(self):
        
        self.fittedGrid, = self.ax.plot(self.pts[0], self.pts[1], "o", picker=5)
        d = self.d
        self.vertlines = self.ax.plot(self.pts[0].reshape((2 * d, 2 * d)), self.pts[1].reshape((2 * d, 2 * d)))
        self.horzlines = self.ax.plot(self.pts[0].reshape((2 * d, 2 * d)).transpose(), self.pts[1].reshape((2 * d, 2 * d)).transpose())



    def error(self, vector):
        if self.focalLength is None:
            the_matrix = mat_from_vec(vector)
        else:
            the_matrix = mat_from_vec(np.concatenate([vector, [self.focalLength]]))
        the_pts = projectPoints(self.grid, the_matrix, self.image)
        mask = self.gridOnScreen != 0
        rotation = vector[3:6]
        if self.focalLength:
            focal = 1
        else:
            focal = 1/vector[6]
        error = np.sum(((self.gridOnScreen[mask] - the_pts[mask]))**2)


        
        return error



    def error_active_params(self, vector):
        the_vector = self.vector.copy()
        the_vector[self.activeParams] = vector

        the_matrix = mat_from_vec(the_vector)

        the_pts = projectPoints(self.grid, the_matrix, self.image)
        mask = self.gridOnScreen != 0
        rotation = the_vector[3:6]
        if self.focalLength:
            focal = 1
        else:
            focal = 1/the_vector[6]
        error = np.sum((self.gridOnScreen[mask] - the_pts[mask])**2) 

        #print(self.gridOnScreen)
        #print("==========")
        #print(the_vector)
        if 0:
            error2 = 0
            for world_point, screen_point in zip(self.grid.transpose(), self.gridOnScreen.transpose()):
                if screen_point[0] != 0:
                    print(world_point)
                    print(screen_point)
                    error2 += error_point_compiled(world_point, screen_point, the_vector, self.image)
            #print (error, error2)
        
        return error


    def get_best_vector(self):
        if self.focalLength is None:
            res = scipy.optimize.minimize(self.error, self.vector.copy())
        
            return res.x
        res = scipy.optimize.minimize(self.error, np.array([0, 0, -5, 0, 0, 0]))
        print("fixed_fl")

        return np.concatenate([res.x, [self.focalLength]])

    def get_best_vector_active_params(self):

        res = scipy.optimize.minimize(self.error_active_params, self.vector.copy()[self.activeParams])
        
        return res.x

    def onpick(self, event):    
        ind = event.ind
        if self.activeIndex != ind:
            self.is_pick = True
    
        self.activeIndex = ind[0]

    def process_updated_annotation(self):
        self.trueGrid.set_data(self.gridOnScreen[0], self.gridOnScreen[1])
        self.fig.canvas.draw()
        
        #self.vector = self.get_best_vector()
        mymatrix = mat_from_vec(self.vector)
        
        self.pts = projectPoints(self.grid, mymatrix, self.image)
        d = self.d
        self.fittedGrid.set_data(self.pts[0], self.pts[1])
        list(map(lambda data: data[0].set_data(data[1], data[2]), zip(self.vertlines, self.pts[0].reshape((2 * d, 2 * d)), self.pts[1].reshape((2 * d, 2 * d)))))
        list(map(lambda data: data[0].set_data(data[1], data[2]), zip(self.horzlines, self.pts[0].reshape((2 * d, 2 * d)).transpose(), self.pts[1].reshape((2 * d, 2 * d)).transpose())))
        #self.horzlines.set_data(self.pts[0].reshape((2 * d, 2 * d)).transpose(), self.pts[1].reshape((2 * d, 2 * d)).transpose())
        self.fig.canvas.draw()

    def on_click(self, event):
       
        if self.is_pick:
            self.is_pick = False
            return
        self.gridOnScreen[:, self.activeIndex] = event.xdata, event.ydata
        self.process_updated_annotation()

    def on_key_press(self, event):
        
        if event.key == 'd':
            self.gridOnScreen[:, self.activeIndex] = 0, 0
            self.process_updated_annotation()
        

import pickle
import click
import os
@click.command()
@click.argument('folder')
@click.argument('outfolder')
def manual_align(folder, outfolder):
    for fname in sorted(os.listdir(folder)):
        full_name = os.path.join(folder, fname)
        print(full_name)

        if full_name[-4:] in ['.jpg', '.png']:
            t2 = np.array(Image.open(full_name))
            result = ManualAligner(t2)
            result.interactive_align()
            print(result)
            with open(os.path.join(outfolder, fname + ".pickle"), "wb") as dump:
                pickle.dump((result.vector, result.gridOnScreen), dump)



if __name__ == "__main__":
    manual_align()
        






