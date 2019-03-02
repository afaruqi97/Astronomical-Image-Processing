#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 09:15:23 2019

A1 - Astronomical Image Processing 
Third Year Labs

by Amena Faruqi and Sophia Zomerdijk-Russell

"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import time

hdulist = fits.open("clean_mosaic.fits")
image_data = hdulist[0].data

start = time.time()        
print("y = ", image_data.shape[0])
print("x = ", image_data.shape[1])

# --------------- Remove stars/blooms/noise from data -------------------

def cleanup(image_data):

    for y in range(image_data.shape[0] ): 
        for x in range(image_data.shape[1]):
            
            # (1) Main Star
            r1 = ((x-1430)**2 + (y-3200)**2)**0.5
            # (2) Top bright one on left of (1)
            r2 = ((x-1315)**2 + (y-4397)**2)**0.5
            # (3) Second bright one below and to right of (2)
            r3 = ((x-1365)**2 + (y-4332)**2)**0.5
            # (4) To left of (2) and (3)
            r4 = ((x-560)**2 + (y-4098)**2)**0.5
            # (5) Slightly to right of (1)
            r5 = ((x-1461)**2 + (y-4032)**2)**0.5
            # (6) Big one to right
            r6 = ((x-2133)**2 + (y-3760)**2)**0.5
            # (7) Far right below (6)
            r7 = ((x-2465)**2 + (y-3417)**2)**0.5
            # (8) Big one to left next to (1)
            r8 = ((x-779)**2 + (y-3321)**2)**0.5
            # (9) Big one slightly below to left of (1)
            r9 = ((x-976)**2 + (y-2773)**2)**0.5
            # (10) Below (9)
            r10 = ((x-905)**2 + (y-2289)**2)**0.5
            #(11) Opposite (10) on right
            r11 = ((x-2131)**2 + (y-2309)**2)**0.5
            # (12) Bottom right
            r12 = ((x-2089)**2 + (y-1424)**2)**0.5
            
            if r1 < 320:
                image_data[y,x] = 0 
            elif r2 < 25:
                image_data[y,x] = 0
            elif r3 < 23:
                image_data[y,x] = 0
            elif r4 < 33:
                image_data[y,x] = 0
            elif r5 < 27:
                image_data[y,x] = 0
            elif r6 < 52:
                image_data[y,x] = 0
            elif r7 < 33:
                image_data[y,x] = 0
            elif r8 < 62:
                image_data[y,x] = 0
            elif r9 < 48:
                image_data[y,x] = 0
            elif r10 < 47:
                image_data[y,x] = 0
            elif r11 < 34:
                image_data[y,x] = 0
            elif r12 < 32:
                image_data[y,x] = 0
            else:
                pass
    
    # (1) Main star vertical bloom
    image_data[0:4611, 1425:1455] = 0
    # (1) Main star horizontal blooms
    image_data[115:150, 1290:1535] = 0 # (a)
    image_data[218:267, 1391:1474] = 0 # (b)
    image_data[312:360, 1015:1711] = 0 # (c)
    image_data[425:480, 1100:1653] = 0 # (d)
    
    image_data[422:454,1025:1045] = 0 #cross next to blooms
    
    # (8) bloom
    image_data[3204:3420, 772:779] = 0
    
    # (9) bloom
    image_data[2703:2837, 969:977] = 0
    
    # (10) bloom
    image_data[2223:2356, 901:908] = 0
    
    #Noise in top right corner
    image_data[-220:,-410:] = 0
    image_data[-410:,-210:] = 0
    
    # Cutting off edges to reduce noise:
    image_data = np.delete(image_data, np.s_[0:100], 0)
    image_data = np.delete(image_data, np.s_[-100:], 0)
    image_data = np.delete(image_data, np.s_[0:100], 1)
    image_data = np.delete(image_data, np.s_[-100:], 1)
    
                
    print("max after =", np.max(image_data))  
    hdulist[0].data = image_data
    plt.figure()   
    hdulist.writeto('clean_mosaic.fits')   # save clean FITS file 
    plt.imshow(image_data)
    end = time.time()
    print(end-start)
    return image_data


# ------------ Make histogram of background pixel values -----------------

def histogram(data):
    mu = np.round(np.mean(data),1)
    std = np.round(np.std(data),1)
    print(mu,std)
    def gaussian(mu,std,data):
        return (np.exp(-((data-mu)**2)/(2*std**2)))/(np.sqrt(2*np.pi)*std)
        
    plt.figure()
    x = np.linspace(np.min(data), np.max(data), 100)
    plt.plot(x, gaussian(mu,std,x), 'k', label = 'Gaussian fit: $\mu$ = %s, $\sigma$ = %s' % (mu, std))
    plt.hist(data,bins=108, color='g', normed = True, label = 'Histogram of pixel values')
    plt.title('Dsitribution of Background Pixel Values')
    plt.legend(loc = 'upper left')
    plt.xlabel('Pixel Value')
    plt.xlim(3360,3465)
    plt.ylim(0,0.048)
    plt.savefig('bkg_histogram2.jpg', dpi=250)
    plt.show()

image_data = cleanup(image_data)
flat_data = image_data.flatten()
flat_data = [d for d in flat_data if 3350 < d < 3460] # cut off long tail of Gaussian
histogram(flat_data)


