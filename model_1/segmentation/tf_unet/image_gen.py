from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
from tf_unet.image_util import BaseDataProvider
from tifffile import imsave

class GrayScaleDataProvider(BaseDataProvider):
    channels = 1
    n_class = 2
    
    def __init__(self, nx, ny, **kwargs):
        super(GrayScaleDataProvider, self).__init__()
        self.nx = nx
        self.ny = ny
        self.kwargs = kwargs
        rect = kwargs.get("rectangles", False)
        if rect:
            self.n_class=3
        
    def _next_data(self):
        return create_image_and_label(self.nx, self.ny, **self.kwargs)

class RgbDataProvider(BaseDataProvider):
    channels = 3
    n_class = 2
    
    def __init__(self, nx, ny, **kwargs):
        super(RgbDataProvider, self).__init__()
        self.nx = nx
        self.ny = ny
        self.kwargs = kwargs
        rect = kwargs.get("rectangles", False)
        if rect:
            self.n_class=3

        
    def _next_data(self):
        data, label = create_image_and_label(self.nx, self.ny, **self.kwargs)
        return to_rgb(data), label

def MakeNonoverlappingCircles(nx, ny, border, r_min, r_max, currentCircles):

    while True:    
        a = np.random.randint(border, nx-border)
        b = np.random.randint(border, ny-border)
        c = np.random.randint(r_min, r_max)
        noOverlap = True
        for x, y, r in currentCircles:
            if ((x - a)*(x - a) + (y - b)*(y - b) <= (r + c)*(r + c)):
                noOverlap = False
                break
        if noOverlap:
            return a, b, c

def MakeNonoverlappingRectangles(nx, ny, r_min, r_max, currentCircles):

    while True:
        a = np.random.randint(nx)
        b = np.random.randint(ny)
        c =  np.random.randint(r_min, r_max)
            
        noOverlap = True
        for x, y, r in currentCircles:
            if ((x - (a + c/2))*(x - (a + c/2)) + (y - (b + c/2))*(y - (b + c/2)) <= (r + c/np.sqrt(2))*(r + c/np.sqrt(2))):
                noOverlap = False
                break
        if noOverlap:
            return a, b, c


def create_image_and_label(nx,ny, cnt = 10, r_min = 5, r_max = 50, border = 10, sigma = 20, rectangles=False, ringFlag=False):
        
    image = np.ones((nx, ny, 1))
    label = np.zeros((nx, ny, 3), dtype=np.bool)
    mask = np.zeros((nx, ny), dtype=np.bool)
    circles = []
    # increase min radius if only rings are used
    if ringFlag:
        r_min = 30
        
        for _ in range(cnt):
            a, b, r = MakeNonoverlappingCircles(nx, ny, border, r_min, r_max, circles)  
            h = np.random.randint(1,255)

            # append to circles
            circles.append((a, b, r))

            y,x = np.ogrid[-a:nx-a, -b:ny-b]
            m = np.abs(x*x + y*y - r*r) <= 400

            # label mask
            n = np.abs(x*x + y*y - r*r) <= 400
            mask = np.logical_or(mask, n)

            image[m] = h
    else:        
        for _ in range(cnt):
            a = np.random.randint(border, nx-border)
            b = np.random.randint(border, ny-border)
            r = np.random.randint(r_min, r_max)
            h = np.random.randint(60,80)

            # append to circles
            circles.append((a, b, r))

            y,x = np.ogrid[-a:nx-a, -b:ny-b]
            m = x*x + y*y <= r*r

            mask = np.logical_or(mask, m)

            image[m] = h

    label[mask, 1] = 1
    
    if rectangles:
        rmin = 20
        mask = np.zeros((nx, ny), dtype=np.bool)
        for _ in range(cnt):
            a, b, r = MakeNonoverlappingRectangles(nx, ny, r_min, r_max, circles)
            h = 220#np.random.randint(1,255)
    
            m = np.zeros((nx, ny), dtype=np.bool)
            m[a:a+r, b:b+r] = True
            mask = np.logical_or(mask, m)
            image[m] = h
            
        label[mask, 2] = 1
        
        label[..., 0] = ~(np.logical_or(label[...,1], label[...,2]))
    
    image += np.random.normal(scale=sigma, size=image.shape)
    image -= np.amin(image)
    image /= np.amax(image)

    if rectangles:
        return image, label
    else:
        return image, label[..., 1]




def to_rgb(img):
    img = img.reshape(img.shape[0], img.shape[1])
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    blue = np.clip(4*(0.75-img), 0, 1)
    red  = np.clip(4*(img-0.25), 0, 1)
    green= np.clip(44*np.fabs(img-0.5)-1., 0, 1)
    rgb = np.stack((red, green, blue), axis=2)
    return rgb

