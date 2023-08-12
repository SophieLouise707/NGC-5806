#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:40:42 2023

@author: Sophieslaptop
"""

import glob
from os import path

import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import ndimage
import numpy as np

from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util
import time
from time import perf_counter as clock


#from sys import exit
import shutil

#Fit ppxf of single spectrum
#Input data header, spectrum, error spectrum, and high_redshift=True/False
#Using xshooter data release 3 templates

#Just stellar fit, with Vazdekis templates
#For stellar pop

#Fit ppxf of single spectrum
#Input data spectrum, error spectrum, and high_redshift=True/False
#For single spec, gal_lin=hdu[0].data

def ppxf_starfit_vazdekis(gal_lin,err,high_redshift=False):
    #Define wavelength array:
    nz = 3682 
    lam1 = ((h1['cd3_3'] * (np.arange(0, nz, 1))) + h1['crval3']) 

    #Cut end of array, where residuals make spectrum too noisy
    wave_lim1 = np.where(lam1<8800)
    lam1 = lam1[wave_lim1]
    gal_lin = gal_lin[wave_lim1]
    err = err[wave_lim1]

    #Cut very beginning of array, which is noisy
    wave_lim2 = np.where(lam1>4800)
    lam1 = lam1[wave_lim2]
    gal_lin = gal_lin[wave_lim2]
    err = err[wave_lim2]
    
    #Can measure by finding FWHM of a sky line--look for one between HA and HB
    #FWHM_gal = 2.65 #FWHM for MUSE. Should double-check this -SR
    FWHM_gal = 2.52

    # If high_redshift is True I pretend the SAURON spectrum was observed at
    # high redshift z0 ~ 1.23. For this I have to broaden both the wavelength
    # range and the instrumental resolution (in wavelength units). You should
    # comment the following three lines if your spectrum was observed at high
    # redshift, and you did not already de-redshift it.
    #
    if high_redshift:
        redshift_0 = 1.23
        lam1 *= 1 + redshift_0
        FWHM_gal *= 1 + redshift_0

    # If the galaxy is at significant redshift, it is easier to bring the
    # galaxy spectrum roughly to the rest-frame wavelength, before calling pPXF
    # (See Sec.2.4 of Cappellari 2017). In practice there is no need to modify
    # the spectrum in any way, given that a red shift corresponds to a linear
    # shift of the log-rebinned spectrum. One just needs to compute the
    # wavelength range in the rest-frame and adjust the instrumental resolution
    # of the galaxy observations.
    #
    if high_redshift:                   # Use these lines if your spectrum is at high-z
        redshift_0 = 1.233              # Initial guess of the galaxy redshift
        lam1 /= 1 + redshift_0     # Compute approximate restframe wavelength range
        FWHM_gal /= 1 + redshift_0      # Adjust resolution in wavelength units
        redshift = 0                    # As I de-redshifted the spectrum, the guess becomes z=0
    else:                               # Use these lines if your spectrum is at low-z
        redshift_0 = 0                  # Ignore cosmological redshift for local galaxies
        redshift = 0.00449               # Initial redshift estimate of the galaxy

    print(len(lam1),len(gal_lin))
    galaxy, ln_lam1, velscale = util.log_rebin(lam1, gal_lin)
    galaxy_norm = galaxy/np.median(galaxy)  # Normalize spectrum to avoid numerical issues
    galaxy_med = np.median(galaxy)
    
    noise=err
    noise_norm = err/galaxy_med #Using noise from data cube -SR           

    # Read the list of filenames from the E-Miles Single Stellar Population
    # library by Vazdekis (2016, MNRAS, 463, 3409) http://miles.iac.es/.
    # A subset of the library is included for this example with permission
    
    ppxf_dir = path.dirname(path.realpath(util.__file__))
    vazdekis = glob.glob(ppxf_dir + '/miles_models/Eun1.30*.fits')
    FWHM_tem = 2.51     # Vazdekis+16 spectra have a constant resolution FWHM of 2.51A.
    velscale_ratio = 1  # Could be 2 to adopt 2x higher spectral sampling for templates than for galaxy

    # Extract the wavelength range and logarithmically rebin one spectrum to a
    # velocity scale 2x smaller than the SAURON galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.
    #
    hdu = fits.open(vazdekis[0])
    ssp = hdu[0].data
    h2 = hdu[0].header
    lam2 = h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1'])

    # The E-Miles templates span a large wavelength range. To save some
    # computation time I truncate the spectra to a similar range as the galaxy.
    good_lam = (lam2 > np.min(lam1)/1.02) & (lam2 < np.max(lam1)*1.02)
    ssp, lam2 = ssp[good_lam], lam2[good_lam]

    lamRange2 = [np.min(lam2), np.max(lam2)]
    sspNew, ln_lam2 = util.log_rebin(lamRange2, ssp, velscale=velscale/velscale_ratio)[:2]
    templates = np.empty((sspNew.size, len(vazdekis)))
    reg_dim = templates.shape[1:]
    print("reg_dim: ", reg_dim)

    # Convolve the whole Vazdekis library of spectral templates
    # with the quadratic difference between the SAURON and the
    # Vazdekis instrumental resolution. Logarithmically rebin
    # and store each template as a column in the array TEMPLATES.

    # Quadratic sigma difference in pixels Vazdekis --> SAURON
    # The formula below is rigorously valid if the shapes of the
    # instrumental spectral profiles are well approximated by Gaussians.
    #
    FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
    sigma = FWHM_dif/2.355/h2['CDELT1']  # Sigma difference in pixels

    for j, file in enumerate(vazdekis):
        hdu = fits.open(file)
        ssp = hdu[0].data
        ssp = ndimage.gaussian_filter1d(ssp[good_lam], sigma)
        sspNew = util.log_rebin(lamRange2, ssp, velscale=velscale/velscale_ratio)[0]
        templates[:, j] = sspNew/np.median(sspNew[sspNew > 0])  # Normalizes templates

    #Determine mask, using version of determine_goodpixels that masks emission lines and sky
    goodPixels = util.determine_goodpixels(ln_lam1, lamRange2, redshift)
    
    # Here the actual fit starts. The best fit is plotted on the screen.
    # Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
    #
    c = 299792.458
    vel = c*np.log(1 + redshift)   # eq.(8) of Cappellari (2017, MNRAS)
    start = [vel, 200.]  # (km/s), starting guess for [V, sigma]
    t = clock()


    #moments=4: velocity, vel disp, h3, h4
    #Run ppxf on just stellar continuum, to get stellar velocity and dispersion
    plt.figure(figsize=(15,8))
    pp = ppxf(templates, galaxy_norm, noise_norm, velscale, start,
                plot=True, moments=4, goodpixels=goodPixels,
                lam=np.exp(ln_lam1), lam_temp=np.exp(ln_lam2),
                degree=4, velscale_ratio=velscale_ratio)
    #print("velscale = ", velscale)
    
    #Save values found by ppxf
    star_params = pp.sol
    star_err = pp.error
    star_chi2 = pp.chi2
    star_bestfit = pp.bestfit
    weights = pp.weights

    # The updated best-fitting redshift is given by the following
    # lines (using equation 8 of Cappellari 2017, MNRAS)
    vcosm = c*np.log(1 + redshift_0)            # This is the initial redshift estimate
    vpec = pp.sol[0]                            # This is the fitted residual velocity
    vtot = vcosm + vpec                        # I add the two velocities before computing z
    #vtot is what to correct for
    print("test val:",vpec-vcosm,vpec,vcosm)
    redshift_best = np.exp(vtot/c) - 1          # eq.(8) Cappellari (2017)
    errors = pp.error*np.sqrt(pp.chi2)          # Assume the fit is good
    redshift_err = np.exp(vtot/c)*errors[0]/c   # Error propagation

    print("Formal errors:")
    print("     dV    dsigma   dh3      dh4")
    print("".join("%8.2g" % f for f in errors))
    print('Elapsed time in pPXF: %.2f s' % (clock() - t))
    print(f"Best-fitting redshift z = {redshift_best:#.7f} "
        f"+/- {redshift_err:#.2}")
    print("##### END STELLAR FIT ######")
    
#End stellar fit

    return(lam1,star_params,star_bestfit,weights,galaxy_med,star_err,star_chi2)


#Run multi-spec fit on full binned cube

#Read in binned cube file
folder = '/Users/Sophieslaptop/Desktop/NGC_5806/Binned cube files/Binned_SNR50/'
file = folder + 'Binned_cube_SNR50.fits'
#folder = '/Users/Sophieslaptop/Desktop/NGC_5806/Binned cube files/Center16_SNR50/'
#file = folder + 'Center16_binned_SNR50.fits'
hdu = fits.open(file)
spec_cube = hdu[0].data
err_cube = hdu[1].data
h1 = hdu[0].header

start_time = time.time() #Start timer
nz,ny,nx = spec_cube.shape

#Read in flat x and y array, for finding bin locations
#Created when binning data file
flatx,flaty = np.loadtxt(folder + 'flatxy_arrays_SNR50.txt')

flatx = flatx.astype(int)
flaty = flaty.astype(int)

#Read in bin_number, x_bar,y_bar
#binnum_path = '/Users/Sophieslaptop/Desktop/NGC_5806/Binned cube files/Binned_SNR50/bin_number_SNR50.txt'
#coords_path = '/Users/Sophieslaptop/Desktop/NGC_5806/Binned cube files/Binned_SNR50/xy_coords_SNR50.txt'
binnum_path = folder + 'bin_number_SNR50.txt'
coords_path = folder + 'xy_coords_SNR50.txt'
#coords_path = folder + 'center16_coords_SNR50.txt'

bin_number = np.loadtxt(binnum_path)
bin_number = bin_number.astype(int)

print("Bin_number shape: ",bin_number.shape)
print("Flatx shape: ",flatx.shape)


#Find how many bins there are
num_bins = np.max(bin_number)
num_bins = int(num_bins) +1
print("Number of bins: ",num_bins)

x_bar,y_bar = np.loadtxt(coords_path)
x_bar = x_bar.astype(int)
y_bar = y_bar.astype(int)

print("Max y_bar:",max(y_bar))

#Make empty cubes to hold results
#star vel, star disp, gas vel, gas disp, 7 gas fluxes, 7 gas errs
results_cube = np.empty(shape=(6,ny,nx)) #Cube for ppxf outputs #Vaz--will need to change this
bestfit_cube = np.empty(shape=(3200,ny,nx)) #Cube to save ppxf bestfit for each pix
gas_cube = np.empty(shape=(3200,ny,nx))
weights_cube = np.empty(shape=(150,ny,nx)) #Cube to save weight of each stellar template #Vaz-changed
print(bestfit_cube.shape)
#star_bestfit_cube = np.empty(shape=(3200,ny,nx)) #Cube to save stellar component of ppxf bestfit for each pix
#residuals_cube = np.empty(shape=(3200,ny,nx))

i=0
#i = bin number
while (i < num_bins):
    print('COUNT NUMBER:',i)
    #Find x and y coords of bin i's luminosity-weighted centroid
    x = x_bar[i]
    y = y_bar[i]
    
    #get spectrum and error for the bin
    gal_lin = spec_cube[:,y,x]
    err = err_cube[:,y,x]
    
    
    #Run ppxf on bin's spec
    lam1,star_params,star_bestfit,weights,galaxy_med,star_err,star_chi2 = ppxf_starfit_vazdekis(gal_lin,err)
    

    
    #Just once, save lam1 as a txt file, for future plots
    if i==0:
        np.savetxt("Run8_lam1_080723.txt",lam1)
        
        
    star_vel = star_params[0]
    star_disp = star_params[1]
    
    star_vel_err = star_err[0]
    star_disp_err = star_err[1]
    
    #Find index of bin i in the bin_number array
    bin_ind = np.where(bin_number==i)
    #Now find coords of pix in bin
    
    x_coords = flatx[bin_ind]
    y_coords = flaty[bin_ind]
    
    #Save results
    results_cube[0,y_coords,x_coords] = star_vel 
    results_cube[1,y_coords,x_coords] = star_disp
    results_cube[2,y_coords,x_coords] = star_vel_err #Stellar velocity error
    results_cube[3,y_coords,x_coords] = star_disp_err #Stellar velocity dispersion error\
    results_cube[4,y_coords,x_coords] = galaxy_med #Galaxy median, for de-normalizing if needed
    results_cube[5,y_coords,x_coords] = star_chi2 # chi^2 value for the stellar fit
    
    #print(bestfit_pix.shape)
    #print(bestfit_cube[:,y_coords,x_coords].shape)
    #print(x_coords,y_coords)
    
    #residuals_pix = galaxy - bestfit_pix
    
    for j in range(len(y_coords)):
        bestfit_cube[:,int(y_coords[j]),int(x_coords[j])] = star_bestfit
        weights_cube[:,int(y_coords[j]),int(x_coords[j])] = weights
        #star_bestfit_cube[:,int(y_coords[j]),int(x_coords[j])] = star_bestfit_pix
        #residuals_cube[:,int(y_coords[j]),int(x_coords[j])] = residuals_pix
    #Save a residuals cube by doing data-bestfit

    
    i+=1
    
#Print runtime for full cell
end_time = time.time()
runtime = end_time-start_time
print("Total runtime: ",runtime/3600., "hours")

SNR = str(50) #input SNR of vorbin cube

weights_hdu = fits.PrimaryHDU(weights_cube)
weights_file = "Run9_Weights_cube_SNR"+SNR+"_080723.fits"
weights_hdu.writeto(weights_file,overwrite=True)
print("Saved weights to ", weights_file)

results_hdu = fits.PrimaryHDU(results_cube)
resultfile_name = "Run9_Results_cube_SNR"+SNR+"_080723.fits"
results_hdu.writeto(resultfile_name,overwrite=True)
print("Saved results cube to ", resultfile_name)

bestfit_hdu = fits.PrimaryHDU(bestfit_cube)
bestfit_file = "Run9_StarBestfit_cube_SNR"+SNR+"_080723.fits"
bestfit_hdu.writeto(bestfit_file,overwrite=True)
print("Saved bestfit cube to ", bestfit_file)


folder = '/Users/Sophieslaptop/Desktop/NGC_5806/Binned cube files/Binned_SNR50/'

shutil.move('/Users/Sophieslaptop/'+resultfile_name,folder+resultfile_name)
shutil.move('/Users/Sophieslaptop/'+bestfit_file,folder+bestfit_file)
shutil.move('/Users/Sophieslaptop/'+weights_file,folder+weights_file)
print("Moved files to ", folder)
        

