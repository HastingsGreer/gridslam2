import keras
model = keras.models.load_model("line detection\\mediocre_linefinder")

patch_size=128

    
import random
import pickle
import matplotlib.pyplot as plt
import numpy as np
import cv2
def process(test, liveCV=False, graph=False):
    img = np.array(segment3(test, graph=graph))
    gray = img
    if liveCV:
        cv2.imshow('nn',img)
    #edges = np.logical_or(img[2:, 1:-1] != img[:-2, 1:-1],  img[1:-1, :-2] != img[1:-1, 2:])
    from skimage import img_as_bool, io, color, morphology
    import matplotlib.pyplot as plt

    edges = morphology.skeletonize(img)
    if graph:
        plt.imshow(edges)
        plt.show()
    edges = edges.astype(np.uint8)

    img = np.array(test)

    lines = cv2.HoughLines(edges,1,np.pi/180,40)
    if lines is not None and len(lines) > 25:
        lines = cv2.HoughLines(edges,1,np.pi/180,50)
        #print("too many")
    
    return lines

def cvShow(lines, img):
    
    if lines is not None and len(lines.shape) == 3:
        print(lines.shape)
        for rho,theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 2000*(-b))
            y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b))
            y2 = int(y0 - 2000*(a))

            cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    
    cv2.imshow('nn',img)
    return img
        
def segment3(line, graph=True):
    inp = line
    offset = (patch_size - 100) // 2
    line = np.pad(line, ((offset, offset + patch_size //2 + 100),
                         (offset, offset + patch_size //2 + 100),
                         (0, 0)), "reflect")
    shape = line.shape
    
    i_grid, j_grid = np.mgrid[0:shape[0] - patch_size:128, 0:shape[1]-patch_size:128]
    
    B_shape = i_grid.shape
    
    i_grid = i_grid.flatten()
    j_grid = j_grid.flatten()
    blbatch = np.array([
        line[ci:ci + patch_size, cj:cj + patch_size] for ci, cj in zip(i_grid, j_grid)
    ])
    
    
    res = model.predict(blbatch)
    #classes = res[:, :, 0]
    classes = np.argmax(res, -1)
    classes = classes.reshape((B_shape[0] * B_shape[1], 128, 128))
    
    result = np.zeros(line.shape[:-1])
    
    for i, j, block in zip(i_grid, j_grid, classes):
        offset = (patch_size - 100) // 2
        result[i:i + patch_size, j:j + patch_size] = block
    
    
    result = result[offset:offset + inp.shape[0], offset:offset + inp.shape[1]]
    if graph:
        #relevantRegion = line[ patch_size // 4:np.max(i_grid) + patch_size * 3 // 4, patch_size //4 :np.max(j_grid) + patch_size* 3 // 4]
        plt.imshow(inp , cmap="gray")
        #plt.show()
        plt.imshow(result, alpha=.5)
    
        plt.show()
        print(result.shape, line.shape)
        #plt.show()
        
    return result

def lineFromRhoTheta(rt):
    x = []
    y = []

    if rt is not None:
        for rho,theta in rt[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = (x0 + 800*(-b))
            y1 = (y0 + 800*(a))
            x2 = (x0 - 800*(-b))
            y2 = (y0 - 800*(a))
            x.append([x1, x2])
            y.append([y1, y2])
            
    x = np.array(x)
    y = np.array(y)
    return x, y

from sklearn.cluster import AffinityPropagation, k_means
from sklearn import mixture
def reduce(lines):
    if lines is not None:
        af = AffinityPropagation(preference=-.01)
        af.fit(lines[:, 0] / np.array([[300, 1]]))

        real_lines = af.cluster_centers_ * np.array([[300, 1]])
        return np.expand_dims(real_lines, 1)
    
    
def split(lines):
    if lines is not None:
        idxs = np.arange(len(lines))
        angles = lines[:, 0, 1] * 2
        vangles = np.array([np.cos(angles), np.sin(angles)]).transpose()

        clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
        clf.fit(vangles)
        label = clf.predict(vangles)
        #print(label.shape)
        l1 = lines[label == 0]
        l2 = lines[label == 1]
        #plt.scatter(vangles[label == 0, 0], vangles[label == 0, 1])
        #plt.scatter(vangles[label == 1, 0], vangles[label == 1, 1])
        #plt.show()
        return l1, l2
    return None, None

imgs2 = pickle.load(open("line detection\\testdata", "rb"))
test = random.choice(imgs2)[0]

dialation = 1


def get_lines(img, graph=False, liveCV=False):
	lines = process(img[::dialation, ::dialation], graph=False, liveCV=liveCV)

	l1, l2 = split(reduce(lines))
	if graph:
		#for l in l1, l2:
		#    plt.scatter(l[:, 0, 0], l[:, 0, 1])
		#plt.show()

		for l, color in zip([l1, l2], ["red", "blue"]):
		    xt, yt = lineFromRhoTheta(l)
		    plt.plot(xt.transpose(), yt.transpose(), c=color)
		plt.imshow(img[::dialation, ::dialation])
		plt.show()
	return l1, l2
