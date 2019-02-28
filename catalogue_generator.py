# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:35:52 2019

A1 - Astronomical Image Processing 
Third Year Labs

Amena Faruqi & Sophia Zomerdijk-Russell

"""

from astropy.io import fits,ascii
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import time

hdulist = fits.open("clean_mosaic.fits")
image_data = hdulist[0].data
mask = image_data == 0
masked = np.ma.masked_array(image_data,mask)
ylim,xlim = masked.data.shape   # size of image 
maxp_lim = 3475     # chosen cut-off for galaxy brightness 
bkg_cutoff = 3458   # chosen cut-off for background values (3-sigma away from mean background)

#----------------------- Initialise Data Lists ---------------------------

numgals = []          # galaxy number count
xpos = []             # galaxy x coordinate
ypos = []             # galaxy y coordinate
apsizes = []          # aperture radius
galtypes = []         # galaxy type
counts_list = []      # galaxy counts
counts_errs = []      # counts error
backgrounds = []      # local background around galaxy
magnitudes = []       # magnitude of galaxy
mag_errs = []         # magnitude error

#------------------------ Define Useful Functions --------------------------

def make_plot(magnitudes,mag_errs):
    """
    Generates a plot of galaxy number count against magnitude i.e. the number 
    of galaxies brighter than a given m value. 
    """
    logNs = []      # log (base-10) of number count
    Ns = []         # galaxy number count 
    mags = np.linspace(min(magnitudes) + 0.5 ,max(magnitudes),50) # magnitude bins
    for m in mags:
        logN = np.log10(sum(i < m for i in magnitudes))
        N = sum(i < m for i in magnitudes)
        logNs.append(logN)
        Ns.append(N)

    logN_errs = [1/((np.log(10)*(n**(0.5)))) for n in Ns]
    plt.errorbar(mags,logNs,yerr = logN_errs, fmt = '.', markersize = 3, ecolor = 'r', elinewidth = 0.5, capsize = 2, color = 'k')
    grad = (logNs[30] - logNs[10])/(mags[30] - mags[10])
    print(grad)
    plt.plot(mags[2:45], [(grad*mag) - 1.8 for mag in mags[2:45]], 'g--')
    plt.xlabel('m')
    plt.ylabel('$log_{10}$N(<m)')
    plt.show()


def elliptical_apsize(x0,y0,bk):
    """
    Identifies elliptical galaxies by determining the comparing the radii of a 
    galaxy in 8 different directions. Based on their length to width ratio, galaxies 
    are assigned a type E0-E7, where E7 is very elongated, E0 is circular.
    
    """
    maxp = masked.data[y0,x0]
    x = x0 
    while masked.data[y0,x] > bk and 1 < x < xlim - 1: # radius in +ve x direction
        x += 1
    else:
        if masked.data[y0,x] == 0 or x == xlim - 1:  # if loop hits an edge/deleted star, set radius to zero
            x_rad_upper = 0
        else:
            x_rad_upper = abs(x0 - x)        
        
    y = y0
    while masked.data[y,x0] > bk and 1 < y < ylim - 1: # radius in +ve y direction
        y += 1
    else:
        if masked.data[y,x0] == 0 or y == ylim - 1: # if loop hits an edge/deleted star, set radius to zero
            y_rad_upper = 0
        else:
            y_rad_upper = abs(y - y0)
        
    x = x0 
    while masked.data[y0,x] > bk and 1 < x < xlim - 1:  # radius in -ve x direction
        x -= 1
    else:
        if masked.data[y0,x] == 0 or x == 0:  # if loop hits an edge/deleted star, set radius to zero
            x_rad_lower = 0
        else:
            x_rad_lower = abs(x0 - x) 
        
    y = y0
    while masked.data[y,x0] > bk and 1 < y < ylim - 1:  # radius in -ve y direction
        y -= 1
    else:
        if masked.data[y,x0] == 0 or y == 0:  # if loop hits an edge/deleted star, set radius to zero
            y_rad_lower = 0
        else:
            y_rad_lower = abs(y - y0)
    
    x = x0 
    y = y0 
    while masked.data[y,x] > bk and 1 < y < ylim - 1 and 1 < x < xlim - 1:  # radius in +ve x and y direction (diagonal)
        x += 1
        y += 1
    else:
        if masked.data[y,x] == 0 or x == xlim - 1 or y == ylim - 1: # if loop hits an edge/deleted star, set radius to zero
            upper_right_rad = 0
        else:
            upper_right_rad = ((x-x0)**2 + (y-y0)**2)**0.5
    
    x = x0 
    y = y0 
    while masked.data[y,x] > bk and 1 < y < ylim - 1 and 1 < x < xlim - 1:  # radius in -ve x and y direction (diagonal)
        x -= 1
        y -= 1
    else:
        if masked.data[y,x] == 0 or x == 0 or y == 0: # if loop hits an edge/deleted star, set radius to zero
            lower_left_rad = 0
        else:
            lower_left_rad = ((x-x0)**2 + (y-y0)**2)**0.5
        
    x = x0 
    y = y0 
    while masked.data[y,x] > bk and 1 < y < ylim - 1 and 1 < x < xlim - 1: # radius in +ve x and -ve y direction (diagonal)
        x += 1
        y -= 1
    else:
        if masked.data[y,x] == 0 or x == xlim - 1 or y == 0: # if loop hits an edge/deleted star, set radius to zero
            lower_right_rad = 0
        else:
            lower_right_rad = ((x-x0)**2 + (y-y0)**2)**0.5  
        
    x = x0 
    y = y0 
    while masked.data[y,x] > bk and 1 < y < ylim - 1 and 1 < x < xlim - 1:  # radius in -ve x and +ve y direction (diagonal)
        x -= 1
        y += 1
    else:
        if masked.data[y,x] == 0 or x == 0 or y == ylim - 1:  # if loop hits an edge/deleted star, set radius to zero
            upper_left_rad = 0
        else:
            upper_left_rad = ((x-x0)**2 + (y-y0)**2)**0.5  

    # Selecting the radii in the given direction to use in aperture size calculation/galaxy identification
    if x_rad_upper == 0 or x_rad_lower == 0:             # Use the non-zero value if one is equal to zero
        x_rad = max([x_rad_upper,x_rad_lower])
    else:
        x_rad = (x_rad_upper + x_rad_lower)/2            # Use average value if both are non-zero
   
    if y_rad_upper == 0 or y_rad_lower == 0:             # Use the non-zero value if one is equal to zero
        y_rad = max([y_rad_upper,y_rad_lower])
    else:
        y_rad = (y_rad_upper + y_rad_lower)/2            # Use average value if both are non-zero
   
    if upper_right_rad == 0 or lower_left_rad == 0:      # Use the non-zero value if one is equal to zero
        diag_rad1 = max([upper_right_rad,lower_left_rad])
    else:
        diag_rad1 = (upper_right_rad + lower_left_rad)/2 # Use average value if both are non-zero
   
    if lower_right_rad == 0 or upper_left_rad == 0:      # Use the non-zero value if one is equal to zero
        diag_rad2 = max([lower_right_rad,upper_left_rad])
    else:
        diag_rad2 = (lower_right_rad + upper_left_rad)/2 # Use average value if both are non-zero
    
    a_diag = float(max([diag_rad1,diag_rad2]))    # diagonal major axis
    b_diag = float(min([diag_rad1,diag_rad2]))    # diagonal minor axis
    a_xy = float(max([x_rad,y_rad]))              # x-y major axis 
    b_xy = float(min([x_rad,y_rad]))              # x-y minor axis 
    
    # Defining galaxy type
    if a_diag > 0 and a_xy > 0:      # check a is non-zero (zero for very small/single-pixel galaxies)
        E_diag = int(np.round(10*(1-(b_diag/a_diag))))  # formula for defining galaxy type 
        E_xy = int(np.round(10*(1 - (b_xy/a_xy))))  
        if E_diag > 7 or E_xy > 7:   # correct galaxies types misidentified due to being cut off at an edge
            E_diag = 7
            E_xy = 7
        
        if E_diag > 0 and maxp > maxp_lim:     # diagonally oriented elliptical, maxp limit set for smeared background pixels that looked like ellipticals 
            galtype = 'E%s' %E_diag            # expressing galaxy type in the standard format
      
        elif E_xy > 0 and maxp > maxp_lim:     # vertically or horizontally oriented elliptical,  maxp limit set for smeared background pixels that looked like ellipticals 
            galtype = 'E%s' %E_xy              # expressing galaxy type in the standard format
        else:                                  
            galtype = 'E0'                     # non-elliptical/circular galaxy
            
    else:
        galtype = 'E0'               # assumed tiny galaxies are circular due to low-resolution
        
    apsize = max([x_rad,y_rad,diag_rad1,diag_rad2])  # to block out entire galaxy, use largest radius
    return apsize,galtype


def make_box(x0,y0, boxsize):
    """
    To reduce runtimes, a small box was defined around the location of the maximum pixel (maxp), 
    so a smaller area needed to be scanned through to mask the galaxy and characterise it. This 
    function provides the dimensions of a box of the appropriate size. 
    """
    if y0 - boxsize <= 0 or x0 - boxsize <= 0:               # maxp located close to lower limit in x,y or both
        if y0 - boxsize <= 0 and x0 - boxsize <= 0:          # maxp located close to lower limit in x and y
        # take box from lower left corner of image
            xi = 0                                           
            xf = x0 + boxsize
            yi= 0
            yf = y0 + boxsize
            
        elif y0 - boxsize <= 0 and x0 + boxsize >= xlim:     # maxp located close to lower limit in y and upper limit in x
            # take box from lower right corner of image
            xi = x0 - boxsize
            xf = xlim 
            yi= 0
            yf = y0 + boxsize
            
        elif y0 + boxsize >= ylim and x0 - boxsize <= 0:     # maxp located close to lower limit in x and upper limit in y
            # take box from upper left corner of image
            xi = 0
            xf = x0 + boxsize
            yi= y0 - boxsize
            yf = ylim 
            
        elif y0 - boxsize <= 0:                              # maxp located close to lower limit in y
            # take box from bottom edge of image
            xi = x0 - boxsize
            xf = x0 + boxsize
            yi= 0
            yf = y0 + boxsize
            
        elif x0 - boxsize <= 0:                              # maxp located close to lower limit in x
            # take box from left edge of image 
            xi = 0
            xf = x0 + boxsize
            yi= y0 - boxsize
            yf = y0 + boxsize
            
    elif y0 + boxsize >= ylim or x0 + boxsize >= xlim:       # maxp located close to upper limit in x,y or both
        if y0 + boxsize >= ylim and x0 + boxsize >= xlim:    # maxp located close to upper limit in x and y
            # take box from upper right corner of image 
            xi = x0 - boxsize
            xf = xlim 
            yi= y0 - boxsize
            yf = ylim
            
        elif y0 + boxsize >= ylim:                           # maxp located close to upper limit in y
            # take box from top edge of image
            xi = x0 - boxsize
            xf = x0 + boxsize
            yi= y0 - boxsize
            yf = ylim 
            
        elif x0 + boxsize >= xlim:                           # maxp located close to upper limit in x
            # take box from right edge of image 
            xi = x0 - boxsize
            xf = xlim 
            yi= y0 - boxsize
            yf = y0 + boxsize
    else:                                                   # maxp not close to any edges/limits
        # take box of boxsize around maxp
        xi = x0 - boxsize
        xf = x0 + boxsize
        yi= y0 - boxsize
        yf = y0 + boxsize
    
    return xi,xf,yi,yf

def generate_catalogue():
    st = time.time()
    new_data = masked[~masked.mask].data       # 1D array/list of unmasked values
    numgal = 0                                 # galaxy number count
    maxp = np.amax(new_data)                   # value of brightest pixel
    magzpt = 25.3                              # calibration (from FITS header)
    magzrr = 0.02                              # error on calibration (from FITS header)
    
    while maxp > maxp_lim:
        print(maxp)
        new_data = masked[~masked.mask].data 
        maxp = np.amax(new_data)      # identify brightest pixel
        y0,x0 = np.where(masked.data == maxp)[0][0], np.where(masked.data == maxp)[1][0] # location of brightest pixel
        print(numgal)
        apsize,galtype = elliptical_apsize(x0,y0,bkg_cutoff)   # calculate aperture size and galaxy type for brightest pixel
        upper_apsize = apsize + 10                       # visually chosen upper limit on apsize, used to determine background
        if apsize > 0:    
            if masked.mask[y0,x0] == True:                   # ignore previously masked galaxies 
                masked.data[y0,x0] = 0
            else:
                numgal += 1                                # increase galaxy count by one
                plist = []                                 # list of pixel values within aperture 
                bkg_vals = []                              # list of pixel values between upper aperture and aperture
                masked.mask[y0,x0] = True                  # mask galaxy 
                boxsize = int(apsize + 50)                 # size of box taken around brightest pixel
                xi,xf,yi,yf = make_box(x0,y0, boxsize)     # define box around x0,y0
                for y in range(yi,yf): 
                    for x in range(xi,xf): 
                        r = ((x-x0)**2 + (y-y0)**2)**0.5       # mask circle of radius apsize 
                        if (apsize < r < upper_apsize):
                            bkg_vals.append(masked.data[y,x])  # store background pixel vaues 
                        if r < apsize:
                            plist.append(masked.data[y,x])     # store galaxy pixel values
                            masked.mask[y,x] = True            # mask galaxy contained within apsize
                            masked.data[y,x] = 0
                        
                
                bkg_vals = [b for b in bkg_vals if bkg_cutoff > b > 0]
                bkg = np.mean(bkg_vals)         # determine local background
                       
                plist = [p - bkg for p in plist]  # subtract background values 
                plist = [p for p in plist if p > 0]  # remove negative pixel values (if any)
                counts = np.sum(plist)               # calculate counts value
                c_err = counts**0.5                  # calculate error on counts
                m = magzpt - 2.5*np.log10(counts)    # calculate magnitude
                # calculate magnitude error:
                log_c_err = c_err/(counts*np.log(10))
                m_err = m*((magzrr/magzpt)**2 + (log_c_err/np.log10(counts))**2)**0.5 
                
                # --------------------- Fill Data Lists -------------------------
                
                numgals.append(numgal)
                xpos.append(x0)
                ypos.append(y0)
                apsizes.append(apsize)
                galtypes.append(galtype)
                counts_list.append(counts)
                counts_errs.append(c_err)
                backgrounds.append(bkg)
                magnitudes.append(m)
                mag_errs.append(m_err)
        else:
            masked.data[y0,x0] = 0
   
    print('Runtime: ' , time.time() - st)
    
    hdulist[0].data = image_data # overwrite original FITS data with masked data
    """create table of data values:"""
    data = Table([numgals,xpos,ypos,apsizes,galtypes,counts_list,counts_errs,backgrounds,magnitudes,mag_errs], names = ['galaxy number', 'x', 'y', 'aperture radius', 'galaxy type', 'counts', 'counts error', 'background', 'magnitude', 'magnitude error'])
    ascii.write(data, 'galaxy_catalogue.dat')  # save data table to ASCII catalogue
    hdulist.writeto('final_mosaic.fits')       # save masked FITS file 
   
   
generate_catalogue()
make_plot(magnitudes,mag_errs)

