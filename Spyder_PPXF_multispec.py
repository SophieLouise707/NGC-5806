#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 09:37:22 2023

@author: Sophie Robbins
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

#from sys import exit
from scipy.interpolate import CubicSpline
import shutil

#Fit ppxf of single spectrum
#Input data header, spectrum, error spectrum, and high_redshift=True/False
#Using xshooter data release 3 templates

def ppxf_fit(h1,gal_lin,err,high_redshift=False):
    start_time = time.time() #Start clock, to measure how long it takes to run function
    
    #Define wavelength array:
    nz = 3682 
    lam1 = ((h1['cd3_3'] * (np.arange(0, nz, 1))) + h1['crval3']) 

    redshift = 0.00449
    #Cut end of array, where residuals make spectrum too noisy
    #wave_lim1 = np.where(lam1<9200)
    wave_lim1 = np.where(lam1<8800)
    lam1 = lam1[wave_lim1]
    gal_lin = gal_lin[wave_lim1]
    err = err[wave_lim1]

    #Cut very beginning of array, which is noisy
    wave_lim2 = np.where(lam1>4800)
    #wave_lim2 = np.where(lam1>8000)
    lam1 = lam1[wave_lim2]
    gal_lin = gal_lin[wave_lim2]
    err = err[wave_lim2]
    
    
    
    #Can measure by finding FWHM of a sky line--look for one between HA and HB
    #FWHM_gal = 2.65 #FWHM for MUSE. Should double-check this -SR
    FWHM_gal = 2.52

    redshift_0 = 0
    redshift = 0.00449

    galaxy, ln_lam1, velscale_gal = util.log_rebin(lam1, gal_lin)
    galaxy_norm = galaxy/np.median(galaxy)  # Normalize spectrum to avoid numerical issues
    galaxy_med = np.median(galaxy)
    
    noise=err
    #noise_norm = galaxy_med/SN #Using noise from data cube -SR 
    #noise_norm = noise/np.median(noise)    
    noise_norm = noise/galaxy_med      

    # Reading in stellar templates from the XShooter library
    
    ppxf_dir = path.dirname(path.realpath(util.__file__))
    xshooter_temps = glob.glob(ppxf_dir+'/XShooter_spectra/XSL_DR3_release/XSL_DR3_release/*merged.fits')
    FWHM_tem = 0.598     # 
    velscale_ratio = 1  # could be 2 to adopt 2x higher spectral sampling for templates than for galaxy

    # Extract the wavelength range and logarithmically rebin one spectrum to a
    # velocity scale 2x smaller than the SAURON galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.
    hdu = fits.open(xshooter_temps[0])
    table = hdu[1].data
    h2 = hdu[0].header
    ssp = np.squeeze(table['FLUX_DR'])
    wav_ssp = 10.0 * np.squeeze(table['WAVE']) # convert wavelength units from nm to A

    # The templates span a large wavelength range. To save some
    # computation time I truncate the spectra to a similar range as the galaxy.
    #good_lam = np.where((wav_ssp>4600) & (wav_ssp<9000))
    #good_lam = np.where((wav_ssp>7600) & (wav_ssp<9000))
    #Originally 70
    wave_lim = np.where((wav_ssp>(lam1[0]-100.0)) & (wav_ssp<(lam1[-1]+100.0)))
    ssp = ssp[wave_lim]
    wav_ssp = wav_ssp[wave_lim]
    
    # Interpolation of X-shooter templates to a linear walelength scale
    size_interp = np.size(wave_lim)
    new_wav_ssp = np.linspace(wav_ssp[0], wav_ssp[-1], num=size_interp)
    spl = CubicSpline(wav_ssp, ssp)
    ssp_linear = spl(new_wav_ssp)
    
    lamRange2 = [np.min(new_wav_ssp),np.max(new_wav_ssp)]
    
    sspNew, logLam2,velscale = util.log_rebin(lamRange2, ssp_linear, velscale=velscale_gal)
    templates = np.empty((sspNew.size, len(xshooter_temps)))

    
    # Convolve the whole Vazdekis library of spectral templates
    # with the quadratic difference between the SAURON and the
    # Vazdekis instrumental resolution. Logarithmically rebin
    # and store each template as a column in the array TEMPLATES.

    # Quadratic sigma difference in pixels Vazdekis --> SAURON
    # The formula below is rigorously valid if the shapes of the
    # instrumental spectral profiles are well approximated by Gaussians.
    
    FWHM_dif = np.sqrt(FWHM_gal**2 - FWHM_tem**2)
    sigma = FWHM_dif/2.355/(10*h2['SPEC_BIN'])  # Sigma difference in pixels

    j=0
    for file in enumerate(xshooter_temps):
        hdu = fits.open(file[1]) 
        table = hdu[1].data
        ssp = table['FLUX_DR']
        #ssp = table['FLUX']
        wav_ssp = 10.0*np.squeeze(table['WAVE'])
        
        #wav_lim = np.where((wav_ssp>4600) & (wav_ssp<9000))
        #wav_lim = np.where((wav_ssp>7600) & (wav_ssp<9000))
        #Originally lam1[0]+-70
        wave_lim = np.where((wav_ssp>(lam1[0]-100.0)) & (wav_ssp<(lam1[-1]+100.0)))
        lam2 = wav_ssp[wave_lim] 
        ssp = ssp[wave_lim]
        
        #Interpolate all templates
        size_interp = np.size(wave_lim)
        new_wav_ssp = np.linspace(lam2[0], lam2[-1], num=size_interp)
        ssp[~np.isfinite(ssp)] = 0.0
        spl = CubicSpline(lam2, ssp)
        ssp_linear = spl(new_wav_ssp)
       
        #ssp_linear[~np.isfinite(ssp_linear)] = 0.0
        lamRange2 = [np.min(lam2),np.max(lam2)]
        
        ssp_filter = ndimage.gaussian_filter1d(ssp_linear, sigma)

        sspNew, logLam2, velscale = util.log_rebin(lamRange2, ssp_filter, velscale=velscale_gal)
        templates[:, j] = sspNew/np.median(sspNew)  # Normalizes templates
        
        j = j+1
    print(templates.shape)

    print("lamRange2:",lamRange2)
    
    c = 299792.458 #km/s
    
    #dv is the difference in vel between templates and data
    dv = (np.mean(logLam2[:velscale_ratio])- ln_lam1[0])*c 
    #print("dv:",dv)
    vsyst = 1350.0 #km/s
    #vel = c*np.log(1 + redshift)   # eq.(8) of Cappellari (2017, MNRAS)
    #start = [vel, 200.]  # (km/s), starting guess for [V, sigma]
    start = [50.0+dv, 100.]  # (km/s), starting guess for [V, sigma]
    
     #Determine mask, using version of determine_goodpixels that masks emission lines and sky
    goodPixels = util.determine_goodpixels(ln_lam1, lamRange2, 0.00449)
    

    #To add to ppxf call:   #lam=np.exp(ln_lam1), lam_temp=np.exp(ln_lam2),
    
    #print(np.min(lam1))
    #moments=4: velocity, vel disp, h3, h4
    #Run ppxf on just stellar continuum, to get stellar velocity and dispersion
    plt.figure(figsize=(15,8))
    pp = ppxf(templates, galaxy_norm, noise_norm, velscale_gal, start,
                plot=False, moments=2,vsyst=vsyst,goodpixels=goodPixels,
                lam=np.exp(ln_lam1), 
                degree=4, velscale_ratio=velscale_ratio)
    print("Reg_dim = ",templates.shape[1:])
    
    # lam=np.exp(ln_lam1), lam_temp=np.exp(ln_lam2),goodpixels=goodPixels,
    
    #Save values found by ppxf
    star_params = pp.sol
    print("Stellar velocity: ",star_params[0]-dv)
    #star_bestfit = pp.bestfit
    star_weights = pp.weights
    star_err = pp.error
    star_chi2 = pp.chi2

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
    print(f"Best-fitting redshift z = {redshift_best:#.7f} "
        f"+/- {redshift_err:#.2}")
    print("##### END STELLAR FIT ######")
    
#End stellar fit
#####################################################
#Full PPXF fit, with emission lines:
    
    #Define wavelength arrays
    lam1 = np.exp(ln_lam1)
    lam2 = np.exp(logLam2)
    
    #Make array with min and max values for lam1
    lam_minmax = [np.min(lam1)/ (1+redshift),np.max(lam1)/ (1+redshift)] 

    #Find gas templates, gas names, and array of center wavelength for each line
    gas_templates, gas_names, line_wave = util.emission_lines(logLam2, lam_minmax, FWHM_gal)
    #print(gas_names)
    #print(gas_templates.shape)
    #print(line_wave)

    star_vel = star_params[0]
    star_disp = star_params[1]

        
    ngas_comp = 1 #Number of kinematic components for each emission line
    gas_templates = np.tile(gas_templates,ngas_comp)#Shouldn't be needed now, but good to have in case of increasing ngas_comp
    line_wave = np.tile(line_wave,ngas_comp)
    
    optimal_template = templates @ pp.weights #Stellar template
    stars_gas_templates = np.column_stack([optimal_template, gas_templates])
    
    #For now, component 0 = stellar component
    #Component 1 = narrow lines
    #May need broad lines or narrow 2 later, but starting with 1
    component = [0]*1 + [1]*7 #Should be 7 if tie_balmer=False
    #Should be 4 components for stars, but that gave error: needs a template for each component
    gas_component=np.array(component)>0
    moments = [-2,2] #Keeps stellar vel and disp fixed at values found earlier

    #Starting guess for velocity and dispersion
    start = [[star_vel,star_disp],[star_vel,50]] #Stellar vel and disp fixed from earlier run
    
    vlim = lambda x: star_vel + x*np.array([-100, 100])
    bounds = [[vlim(2), [20, 300]],       # Bounds are ignored for the stellar component=0 which has fixed kinematic
              [vlim(2), [1, 100]]]       # I force the narrow component=1 to lie +/-200 km/s from the stellar velocity

    #Define velscale
    d_ln_lam = np.diff(np.log(lam1[[0, -1]]))/(lam1.size - 1)
    velscale = c*d_ln_lam
    velscale = velscale[0]
    
    lamRange2 = [np.min(lam2), np.max(lam2)]
    goodPixels = util.determine_goodpixels_sky(ln_lam1, lamRange2)
    
    plt.figure(figsize=(15,8))
    pp = ppxf(stars_gas_templates, galaxy, noise, velscale,start,vsyst=vsyst,
              plot=1, moments=moments, component=component, goodpixels=goodPixels,
              gas_component=gas_component, gas_names=gas_names,
              lam=lam1,global_search=False,velscale_ratio=velscale_ratio)
    #lam_temp=np.exp(logLam2)

    
    # The updated best-fitting redshift is given by the following
    # lines (using equation 8 of Cappellari 2017, MNRAS)
    vcosm = c*np.log(1 + redshift_0)            # This is the initial redshift estimate
    vpec = pp.sol[0]                            # This is the fitted residual velocity
    vtot = vcosm + vpec                        # I add the two velocities before computing z
    #vtot is what to correct for
    print("test val:",vpec-vcosm,vpec,vcosm)
    redshift_best = np.exp(vtot/c) - 1          # eq.(8) Cappellari (2017)
    print("Best-fitting redshift z =",redshift_best)
    
    vels = pp.sol
    gas_flux = pp.gas_flux
    gas_err = pp.gas_flux_error
    bestfit = pp.bestfit
    gas_fit = pp.gas_bestfit
    gas_chi2 = pp.chi2
    
    #print("Stellar velocity: ",vels[0][0]-dv)
    #print("Gas velocity: ",vels[1][0]-dv)
    
    
    end_time = time.time()
    runtime = end_time-start_time
    print("Spectrum runtime: ",runtime, "seconds")
    return(galaxy,vels,gas_flux,gas_err,bestfit,gas_fit,galaxy_med,dv,star_weights,star_err,lam1,star_chi2,gas_chi2)



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
results_cube = np.empty(shape=(23,ny,nx)) #Cube for ppxf outputs
bestfit_cube = np.empty(shape=(3200,ny,nx)) #Cube to save ppxf bestfit for each pix
gas_cube = np.empty(shape=(3200,ny,nx))
weights_cube = np.empty(shape=(606,ny,nx)) #Cube to save weight of each stellar template 
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
    galaxy,vels_pix, gas_flux_pix,gas_err_pix,bestfit_pix,gas_bestfit,galaxy_med_pix,dv,star_weights,star_err,lam1,star_chi2,gas_chi2 = ppxf_fit(h1,gal_lin,err,high_redshift=False)
    

    
    #Just once, save lam1 as a txt file, for future plots
    if i==0:
        np.savetxt("Run6_lam1_051223.txt",lam1)
        
    star_vels = vels_pix[0]
    gas_vels = vels_pix[1]
    
    star_vel = star_vels[0] - dv
    star_disp = star_vels[1]
    gas_vel = gas_vels[0] - dv
    gas_disp = gas_vels[1]
    
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
    results_cube[2,y_coords,x_coords] = gas_vel
    results_cube[3,y_coords,x_coords] = gas_disp
    
    results_cube[4,y_coords,x_coords] = gas_flux_pix[0] #H beta
    results_cube[5,y_coords,x_coords] = gas_flux_pix[1] #H alpha
    results_cube[6,y_coords,x_coords] = gas_flux_pix[2] #[SII] 6716
    results_cube[7,y_coords,x_coords] = gas_flux_pix[3] #[SII] 6731
    results_cube[8,y_coords,x_coords] = gas_flux_pix[4] #[OIII] 5007
    results_cube[9,y_coords,x_coords] = gas_flux_pix[5] #[OI] 6300d
    results_cube[10,y_coords,x_coords] = gas_flux_pix[6] #[NII] 6583d
    
    results_cube[11,y_coords,x_coords] = gas_err_pix[0] #H beta
    results_cube[12,y_coords,x_coords] = gas_err_pix[1] #H alpha
    results_cube[13,y_coords,x_coords] = gas_err_pix[2] #[SII] 6716
    results_cube[14,y_coords,x_coords] = gas_err_pix[3] #[SII] 6731
    results_cube[15,y_coords,x_coords] = gas_err_pix[4] #[OIII] 5007
    results_cube[16,y_coords,x_coords] = gas_err_pix[5] #[OI] 6300d
    results_cube[17,y_coords,x_coords] = gas_err_pix[6] #[NII] 6583d
    
    results_cube[18,y_coords,x_coords] = galaxy_med_pix #Galaxy median, for de-normalizing if needed
    results_cube[19,y_coords,x_coords] = star_vel_err #Stellar velocity error
    results_cube[20,y_coords,x_coords] = star_disp_err #Stellar velocity dispersion error
    results_cube[21,y_coords,x_coords] = star_chi2 # chi^2 value for the stellar fit
    results_cube[22,y_coords,x_coords] = gas_chi2 #chi^2 value for the full gas fit
    
    #print(bestfit_pix.shape)
    #print(bestfit_cube[:,y_coords,x_coords].shape)
    #print(x_coords,y_coords)
    
    #residuals_pix = galaxy - bestfit_pix
    
    for j in range(len(y_coords)):
        bestfit_cube[:,int(y_coords[j]),int(x_coords[j])] = bestfit_pix
        weights_cube[:,int(y_coords[j]),int(x_coords[j])] = star_weights
        gas_cube[:,int(y_coords[j]),int(x_coords[j])] = gas_bestfit
        #star_bestfit_cube[:,int(y_coords[j]),int(x_coords[j])] = star_bestfit_pix
        #residuals_cube[:,int(y_coords[j]),int(x_coords[j])] = residuals_pix
    #Save a residuals cube by doing data-bestfit

    
    i+=1
    
#Print runtime for full cell
end_time = time.time()
runtime = end_time-start_time
print("Total runtime: ",runtime/3600., "hours")

results_hdu = fits.PrimaryHDU(results_cube)
SNR = str(50) #Input SNR of vorbin cube
resultfile_name = "Run8_Results_cube_SNR"+SNR+"_060223.fits"
results_hdu.writeto(resultfile_name,overwrite=True)

bestfit_hdu = fits.ImageHDU(bestfit_cube)
bestfit_file = "Run8_Bestfit_cube_SNR"+SNR+"_060223.fits"
bestfit_hdu.writeto(bestfit_file,overwrite=True)

weights_hdu = fits.ImageHDU(weights_cube)
weights_file = "Run8_Weights_cube_SNR"+SNR+"_060223.fits"
weights_hdu.writeto(weights_file,overwrite=True)

gas_hdu = fits.ImageHDU(gas_cube)
gas_file = "Run8_GasBestfit_cube_SNR"+SNR+"_060223.fits"
gas_hdu.writeto(gas_file,overwrite=True)

folder = '/Users/Sophieslaptop/Desktop/NGC_5806/Binned cube files/Binned_SNR50/'

shutil.move('/Users/Sophieslaptop/'+resultfile_name,folder+resultfile_name)
shutil.move('/Users/Sophieslaptop/'+bestfit_file,folder+bestfit_file)
shutil.move('/Users/Sophieslaptop/'+weights_file,folder+weights_file)
shutil.move('/Users/Sophieslaptop/'+gas_file,folder+gas_file)
        

