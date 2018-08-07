
# coding: utf-8

# In[4]:


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# In[46]:


d = 3
gridx, gridy = np.mgrid[-d:d:1, -d:d:1]
grid = np.array([gridx.flatten(), gridy.flatten(), np.zeros(((2*d)**2)),np.ones(((2*d)**2)) ])


# In[70]:


#https://www.opengl.org/discussion_boards/showthread.php/197893-View-and-Perspective-matrices

import math
import numpy as np
 
def transform(m, v):
    return np.asarray(m * np.asmatrix(v).T)[:,0]
 
def magnitude(v):
    return math.sqrt(np.sum(v ** 2))
 
def normalize(v):
    m = magnitude(v)
    if m == 0:
        return v
    return v / m
 
def ortho(l, r, b, t, n, f):
    dx = r - l
    dy = t - b
    dz = f - n
    rx = -(r + l) / (r - l)
    ry = -(t + b) / (t - b)
    rz = -(f + n) / (f - n)
    return np.matrix([[2.0/dx,0,0,rx],
                      [0,2.0/dy,0,ry],
                      [0,0,-2.0/dz,rz],
                      [0,0,0,1]])
 
def perspective(fovy, aspect, n, f):
    s = 1.0/math.tan(math.radians(fovy)/2.0)
    sx, sy = s / aspect, s
    zz = (f+n)/(n-f)
    zw = 2*f*n/(n-f)
    return np.matrix([[sx,0,0,0],
                      [0,sy,0,0],
                      [0,0,zz,zw],
                      [0,0,-1,0]])

def simplePerspective(a):
    return np.matrix([[1,0,0,0],
                      [0,1,0,0],
                      [0,0,1,a],
                      [0,0,-a,0]])

def frustum(x0, x1, y0, y1, z0, z1):
    a = (x1+x0)/(x1-x0)
    b = (y1+y0)/(y1-y0)
    c = -(z1+z0)/(z1-z0)
    d = -2*z1*z0/(z1-z0)
    sx = 2*z0/(x1-x0)
    sy = 2*z0/(y1-y0)
    return np.matrix([[sx, 0, a, 0],
                      [ 0,sy, b, 0],
                      [ 0, 0, c, d],
                      [ 0, 0,-1, 0]])
 
def translate(xyz):
    x, y, z = xyz
    return np.matrix([[1,0,0,x],
                      [0,1,0,y],
                      [0,0,1,z],
                      [0,0,0,1]])
 
def scale(xyz):
    x, y, z = xyz
    return np.matrix([[x,0,0,0],
                      [0,y,0,0],
                      [0,0,z,0],
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
 
def lookat(eye, target, up):
    F = target[:3] - eye[:3]
    f = normalize(F)
    U = normalize(up[:3])
    s = np.cross(f, U)
    u = np.cross(s, f)
    M = np.matrix(np.identity(4))
    M[:3,:3] = np.vstack([s,u,-f])
    T = translate(-eye)
    return M * T
 
def viewport(x, y, w, h):
    x, y, w, h = map(float, (x, y, w, h))
    return np.matrix([[w/2, 0  , 0,x+w/2],
                      [0  , h/2, 0,y+h/2],
                      [0  , 0  , 1,    0],
                      [0  , 0  , 0,    1]])


# In[48]:


t1 = np.array(Image.open("test.jpg"))[::4, ::4]
print(t1.shape)
#plt.imshow(t1)
outgrid = grid 
#plt.scatter(outgrid[0], outgrid[1])


# In[49]:


def euler(rotation):
    return rotx(rotation[0])*roty(rotation[1]) * rotz(rotation[2])
def matrix(translation, rotation, focalLength):
    return simplePerspective(focalLength) * euler(rotation) * translate(translation)


# In[63]:


def projectPoints(points, matrix, image):
    out = matrix * points
    out /= out[ 3:4].copy()
    out = out[:2]
    out[0] += image.shape[1] / 2
    out[1] += image.shape[0] / 2
    return np.array(out, dtype=np.float)
    


# In[91]:


mymatrix = matrix((0, 0, -5), (0, 0, 0), 1/1700)
pts = projectPoints(grid, mymatrix, t1)
#plt.imshow(t1)
#plt.scatter(np.array(pts[0]), np.array(pts[1]))


# In[45]:


grid[-1].shape


# In[57]:


(mymatrix * grid)[2]


# In[69]:


pts[0]


# In[85]:


gridOnScreen = np.zeros(grid.shape)


# In[95]:



import numpy as np
import matplotlib.pyplot as plt

print(pts)

fig = plt.figure()

ax = fig.add_subplot(111)
ax.imshow(t1)
ax.set_title('click on points')

line, = ax.plot(pts[0], pts[1], "o", picker=5)  # 5 points tolerance



is_pick = False
def onpick(event):
    global activeIndex
    global is_pick
    is_pick = True
    
    thisline = event.artist
    xdata = thisline.get_xdata()
    ydata = thisline.get_ydata()
    ind = event.ind
    activeIndex = ind[0]
    print(ind[0])
    points = tuple(zip(xdata[ind], ydata[ind]))
    #print('onpick points:', points)
    

fig.canvas.mpl_connect('pick_event', onpick)

def on_click(event):
    global activeIndex
    global is_pick
    if is_pick:
        is_pick = False
        return
    gridOnScreen[activeIndex] = event.xdata, event.ydata
    
fig.canvas.mpl_connect("button_press_event", on_click)



plt.show()

