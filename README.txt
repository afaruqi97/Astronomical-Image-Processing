A1 - Astronomical Image Processing
Amena Faruqi and Sophia Zomerdijk-Russell

Files Needed:
 - A1_mosaic.fits
 - cleanup.py
 - catalogue_generator.py

Please ensure all files are saved in the same directory before running. 

To generate catalogue and masked FITS file:
1) Run cleanup.py, this should generate a new FITS file named 'clean_mosaic.fits' within the same directory.
2) Run catalogue_generator.py, this should generate two files:
   - final_mosaic.fits => FITS file showing all identified galaxies blacked out, can be used to visually assess the 
     completeness of the catalogue
   - galaxy_catalogue.dat => catalogue containing galaxy ID number, (x,y) coordinates of galactic center, radius of 
     aperture, galaxy type, counts, error on counts, local background, magnitude of galaxy, error on magnitude.
     
Total runtime: ~ 15 mins 

