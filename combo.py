import manual_align
import getLines
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np

plt.rcParams["figure.figsize"] = (15, 15)


imgs2 = pickle.load(open("line detection\\testdata", "rb"))

def intersection(line1, line2):
    """Finds the intersection of two lines given in Hesse normal form.

    Returns closest integer pixel locations.
    See https://stackoverflow.com/a/383527/5087436
    """
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([
        [np.cos(theta1), np.sin(theta1)],
        [np.cos(theta2), np.sin(theta2)]
    ])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    #x0, y0 = x0, np.round(y0))
    return [[x0, y0]]


test = random.choice(imgs2)[0]

l1, l2 = getLines.get_lines(test, graph=True)
m = manual_align.ManualAligner(test)
print(m.gridOnScreen.shape)
pts = []
for i, line1 in enumerate(l1):
    for j, line2 in enumerate(l2):
        x = intersection(line1, line2)[0]
        pts.append(x)
        
        #m.gridOnScreen[0, 12 * (5 + i) + (4 + j)] = x[0]
        #m.gridOnScreen[1, 12 * (5 + i) + (4 + j)] = x[1]
pts = np.array(pts)
m.interactive_align()
#m.align_from_saved(m.gridOnScreen)
#m.present_registration()
print(m.gridOnScreen)