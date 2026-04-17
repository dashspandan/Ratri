import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import splrep, splev
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.io import fits
from matplotlib.lines import Line2D
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astroplan import FixedTarget
from astroplan import Observer
from astropy.time import Time
from astroplan.plots import plot_airmass, plot_altitude
import astropy.units as u
import pandas as pd
import urllib.request
import os

def f(M, E, e): # f(E) = E - esin(E) - M
	return E - e*np.sin(E) - M

def Df(E, e): #df/dE = f'(E) = 1 - ecos(E)
	return 1 - e*np.cos(E)

def ecc_anomaly(M, e, tol = 1e-10):
	E0 = M
	fn = f(M, E0, e)
	while max(np.abs(fn)) > tol: #Newton-Raohson method to find root of E - esinE - M = 0
		E_n = E0 - (f(M, E0, e)/Df(E0, e))
		E0 = E_n
		fn = f(M, E0, e)
	return E0

def ecc_RV(e, omega, ph, vsys, Kp, bar_rad_vel): ##find RV(t) for eccentric orbit
	M = 2*np.pi*ph - np.pi/2. - omega #Mean anomalies, 1
	#M = 2*np.pi*ph + np.pi/2. - omega #Mean anomalies, 2
	E = ecc_anomaly(M, e) #Eccentric anomaly
	f = 2*np.arctan((np.sqrt(1+e)/np.sqrt(1-e))*np.tan(E/2.0)) ##true anomaly
	RV_t = vsys - bar_rad_vel + Kp*(np.cos(f + omega) + e*np.cos(omega))
	return RV_t


def splinefit(x, y):
	fn_spl = splrep(x, y, s=0)
	return fn_spl

def broaden(x, y, R, splinefit = False): ##broaden y on the x wavelength grid to a resolution of R
	
	R_actual = 1*np.power(np.power(x[-1]/x[0], 1/x.shape[0]) - 1, -1) ##resolution for regridding
	
	x_mod = np.ones_like(x)
	for n,lam in enumerate(x_mod):
		x_mod[n] = lam*x[0]*np.power((1+(1/R_actual)), n)
	
	fn = splrep(x, y, s=0)
	y_mod = splev(x_mod, fn, der = 0) #fluxes at reggrided wavelengths
	
	convl_res = R
	
	px_res = R_actual/convl_res #ratio between teh FWHMs or resolutions
	
	#dev = px_res/(2*np.sqrt(2*np.log(2))*np.sqrt(np.power(R/inst_res, 2) - 1)) #sigma for kernel
	dev = px_res/(2*np.sqrt(2*np.log(2)))
	
	kernel = Gaussian1DKernel(stddev = dev)
	
	y_con = convolve(y_mod, kernel, normalize_kernel = True, boundary = 'extend') ##convolved transmission spectrum to be multiplied with photon fluxes
	
	if splinefit == True:
		fn_spl = splrep(x_mod, y_con, s=0) ###to be used for the required wavelength grid###
		return x_mod, y_con, fn_spl
	else:
		return x_mod, y_con

def photonfluxearthtop(x, y, d, R_st): #d in parsec, R_s is stellar radius in sun radius
	d_e = d*3.08567758128e+18 ##in cm
	
	R_s = R_st*6.957e+10 ##in cm, sun's radius
	
	lum = y*4*np.pi*R_s**2
	y_p = lum/(4*np.pi*d_e**2) ##distributed over the sphere with earth's radius###
	
	###in terms of number of photons##
	
	h = 6.626196e-27 ##in erg s
	c = 29979245800*1e8 ##in Angstrom/s
	
	ener_grid = (h*c)/x
	
	y_ph = y_p/ener_grid
	
	return y_ph


def signalcalc(inst, modes, exp_time, fn, fn_tr, fn_sky): ##fn is for ither just star or star+planet photon flux at atmosphere top, fn_tr is for telluic absorption
	
	exp_time = exp_time ##in s
	
	if inst == 'crires+':
		tel_rad = 0.5*8.2*1e2 ##in cm, change for each instrument
		area = np.pi*tel_rad**2
	
	if inst == 'carmenes':
		#tel_rad = 0.5*3.5*1e2
		#area = np.pi*tel_rad**2
		area = 9.0*1e4 ##in cm**2
	
	if inst == 'giano':
		#tel_rad = 0.5*3.58*1e2
		area = 9.45*1e4
	
	if inst == 'spirou':
		#tel_rad = 0.5*3.6*1e2
		area = 8.17*1e4
	
	if inst == 'andes':
		tel_rad = 0.5*38.5*1e2
		obs = 0.28**2 #fraction of area obscured by thing at the centre
		area = np.pi*(1-obs)*tel_rad**2 ##total area of telescope available for light
	
	if inst == 'andes_ccd_carmenes':
		tel_rad = 0.5*38.5*1e2
		obs = 0.28**2 #fraction of area obscured by thing at the centre
		area = np.pi*(1-obs)*tel_rad**2 ##total area of telescope available for light
	
	if inst == 'tmt_ccd_giano':
		tel_rad = 0.5*30*1e2 ##in cm, change for each instrument that you add here within another if statement
		area = np.pi*tel_rad**2
	
	if inst == 'nlot_ccd_giano':
		# NLOT collecting area: 121 m^2 (90 segments x 1.35 m^2 each)
		# Slide: National Large Optical-infrared Telescope, Hanle, India
		area = 121 * 1e4  ##in cm^2
	
	if inst == 'tmt':
		tel_rad = 0.5*30*1e2 ##in cm, change for each instrument that you add here within another if statement
		area = np.pi*tel_rad**2
	
	if inst == 'crires+':
		i = 0
		while i < len(modes):
			data = np.load('./instrument_grids_crires/crires+'+modes[i]+'mode.npy')
			if i == 0:
				n_orders = len(data)
				wav = np.zeros((n_orders, len(data[0][0]))) ## 2D cuboid with (n_orders, n_pixels) as dimensions
				sn = np.zeros((n_orders, len(data[0][0])))
				th = np.zeros((n_orders, len(data[0][0])))
				res_el = np.zeros((n_orders, len(data[0][0])))
				j = 0
				while j < n_orders:
					wav[j,:] = data[j][0]
					sn[j,:] = data[j][1]
					th[j,:] = data[j][2]
					res_el[j,:] = data[j][3]
					j = j+1
			if i > 0:
				n_orders = len(data)
				wav_n = np.zeros((n_orders, len(data[0][0])))
				sn_n = np.zeros((n_orders, len(data[0][0])))
				th_n = np.zeros((n_orders, len(data[0][0])))
				res_el_n = np.zeros((n_orders, len(data[0][0])))
				j = 0
				while j < n_orders:
					wav_n[j,:] = data[j][0]
					sn_n[j,:] = data[j][1]
					th_n[j,:] = data[j][2]
					res_el_n[j,:] = data[j][3]
					j = j+1
				
				wav = np.vstack((wav, wav_n))
				sn = np.vstack((sn, sn_n))
				th = np.vstack((th, th_n))
				res_el = np.vstack((res_el, res_el_n))
					
				del wav_n
				del sn_n
				del th_n
				del res_el_n
			i = i+1
	
	if inst == 'spirou' or inst == 'carmenes' or inst == 'giano':
		wav = np.load('./instrument_grids_spirougianocarmenes/wlen_template_'+inst+'.npy')*10 ##nm to angstrom
		th = np.load('./instrument_grids_spirougianocarmenes/eta_'+inst+'.npy')
		if inst == 'giano':
			wav = wav[::-1] ##it is provided in the reverse direction###
			th = th[::-1]
			i = 0
			while i < len(wav):
				wav[i] = wav[i][::-1] ##they are in decreasing instead of increasing order for each order as well
				th[i] = th[i][::-1]
				i = i+1
			
		i = 0
		j = 0
		a,b = wav.shape
		res_el = np.zeros((a,b))
		
		while i < a:
			while j < b:
				if j < b - 1:
					res_el[i,j] = wav[i,j+1] - wav[i,j]
				if j == b - 1:
					res_el[i,j] = res_el[i,j-1]
				j = j+1
			j = 0
			i = i+1
		
		if inst == 'carmenes':
			res_el[:,2039] = res_el[:,2040] ##break between CCDs, put it to the next value for the same CCD
	
	if inst == 'andes':
		i = 0
		while i < len(modes):
			data = np.load('./instrument_grids_andes/parameters_for_ratri_band_'+modes[i]+'_ANDES_new.npy')
			if i == 0:
				wav = data[0]
				th = data[1]
				res_el = data[2]
			if i > 0:
				wav_n = data[0]
				th_n = data[1]
				res_el_n = data[2]
				
				wav = np.append(wav, wav_n)
				th = np.append(th, th_n)
				res_el = np.append(res_el, res_el_n)
				
				del wav_n
				del th_n
				del res_el_n
			i = i+1
	
	if inst == 'andes_ccd_carmenes':
		i = 0
		while i < len(modes):
			data = np.load('./instrument_grids_andes/parameters_for_ratri_band_'+modes[i]+'_ANDES_carmenesinterpol_new.npy')
			if i == 0:
				wav = data[0]
				th = data[1]
				res_el = data[2]
			if i > 0:
				wav_n = data[0]
				th_n = data[1]
				res_el_n = data[2]
				
				wav = np.vstack((wav, wav_n))
				th = np.vstack((th, th_n))
				res_el = np.vstack((res_el, res_el_n))
				
				del wav_n
				del th_n
				del res_el_n
			i = i+1
	
	if inst == 'tmt_ccd_giano':
		wav = np.load('./instrument_grids_spirougianocarmenes/wlen_template_giano.npy')*10 ##nm to angstrom
		th = np.load('./instrument_grids_spirougianocarmenes/eta_giano.npy')
		
		wav = wav[::-1] ##it is provided in the reverse direction###
		th = th[::-1]
		i = 0
		while i < len(wav):
			wav[i] = wav[i][::-1] ##they are in decreasing instead of increasing order for each order as well
			th[i] = th[i][::-1]
			i = i+1
		th = 1*th #increase througput for tmt in comparison to giano, don't
		i = 0
		j = 0
		a,b = wav.shape
		res_el = np.zeros((a,b))
		
		while i < a:
			while j < b:
				if j < b - 1:
					res_el[i,j] = wav[i,j+1] - wav[i,j]
				if j == b - 1:
					res_el[i,j] = res_el[i,j-1]
				j = j+1
			j = 0
			i = i+1
	
	if inst == 'nlot_ccd_giano':
		# NLOT uses GIANO wavelength grid and throughput
		# Throughput scaling TBC pending first-light instrument selection
		wav = np.load('./instrument_grids_spirougianocarmenes/wlen_template_giano.npy')*10 ##nm to angstrom
		th = np.load('./instrument_grids_spirougianocarmenes/eta_giano.npy')
		
		wav = wav[::-1] ##it is provided in the reverse direction###
		th = th[::-1]
		i = 0
		while i < len(wav):
			wav[i] = wav[i][::-1]
			th[i] = th[i][::-1]
			i = i+1
		th = 1*th  # throughput scaling placeholder — update when NLOT instrument specs available
		i = 0
		j = 0
		a,b = wav.shape
		res_el = np.zeros((a,b))
		
		while i < a:
			while j < b:
				if j < b - 1:
					res_el[i,j] = wav[i,j+1] - wav[i,j]
				if j == b - 1:
					res_el[i,j] = res_el[i,j-1]
				j = j+1
			j = 0
			i = i+1
	
	if inst == 'tmt':
		########Generate the common wavelength grid####
		wav_ini = 9500 #in angstrom, for tmt
		wav_fi = 24500 #in angstrom, for tmt
		R = 100000
		arr_wav = [wav_ini]
		i = 1
		while arr_wav[-1] <= wav_fi:
			num = wav_ini*np.power((R/(R-1)),i)
			arr_wav.append(num)
			i = i+1
		
		wav = np.array(arr_wav)
		res_el = wav/R
		th = 0.08*np.ones_like(wav)
	
	mask_1 = th>0.5 #mask for abormally behaving efficiencies
	th[th>0.5] = 0 #set abnormally large valued pixel thoroughputs to 0 (residuals of division of totalth and atmth
	th_m = np.ma.masked_invalid(th) #mask for NaN values
	mask_2 = np.ma.getmask(th_m) ##mask for NaNs
	th = th_m.filled(fill_value = 0) #set these masked values to 0 as well
	
	ph_data_con_fit = splev(wav, fn, der = 0) ##evaluate photon flux at these wavelengths
	tr_con_fit = splev(wav, fn_tr, der = 0) ##atmospheric transmission at these wavelengths
	
	ph_data_det_top = ph_data_con_fit*tr_con_fit
	
	ph_rec_det_eff = ph_data_det_top*exp_time*area*th
	ph_rec_det = ph_rec_det_eff*res_el
	mask_3 = ph_rec_det < 1 ##some negative values and less than 1 photon values will be masked
	ph_rec_det[ph_rec_det < 1] = 1 ##some values (541 total) are found to be negative (don't know why, fitting issues?) set them to 0
	
	if inst == 'carmenes' or 'spirou' or 'giano':
		ph_noise = ph_rec_det**0.5
		sn_met = ph_rec_det/ph_noise
	
	if inst == 'crires+':
		ph_noise = ph_rec_det**0.5
		rod_noise = 10*(7.5**2)
		sn_met = ph_rec_det/np.sqrt(ph_noise**2 + rod_noise)
	
	if inst == 'andes' or 'andes_ccd_carmenes':
		ph_noise = ph_rec_det**0.5
		sky_fl = splev(wav, fn_sky, der = 0)
		ph_data_det_sky_top = sky_fl*tr_con_fit*exp_time*area*th*res_el
		#ph_data_det_sky_top = sky_fl*exp_time*area*th*res_el
		ph_data_det_sky_top[ph_data_det_sky_top < 1] = 1
		sky_noise = ph_data_det_sky_top**0.5
		sn_met = ph_rec_det/np.sqrt(ph_noise**2 + sky_noise**2)
	
	if inst == 'tmt_ccd_giano':
		ph_noise = ph_rec_det**0.5
		sky_fl = splev(wav, fn_sky, der = 0)
		ph_data_det_sky_top = sky_fl*tr_con_fit*exp_time*area*th*res_el
		#ph_data_det_sky_top = sky_fl*exp_time*area*th*res_el
		ph_data_det_sky_top[ph_data_det_sky_top < 1] = 1
		sky_noise = ph_data_det_sky_top**0.5
		rod_noise = 5**2
		sn_met = ph_rec_det/np.sqrt(ph_noise**2 + sky_noise**2+rod_noise)
	
	if inst == 'nlot_ccd_giano':
		# SNR: shot noise + sky + read noise (5e- assumed, TBC)
		ph_noise = ph_rec_det**0.5
		sky_fl = splev(wav, fn_sky, der = 0)
		ph_data_det_sky_top = sky_fl*tr_con_fit*exp_time*area*th*res_el
		ph_data_det_sky_top[ph_data_det_sky_top < 1] = 1
		sky_noise = ph_data_det_sky_top**0.5
		rod_noise = 5**2  # read noise TBC
		sn_met = ph_rec_det/np.sqrt(ph_noise**2 + sky_noise**2 + rod_noise)
	
	if inst == 'tmt':
		ph_noise = ph_rec_det**0.5
		sky_fl = splev(wav, fn_sky, der = 0)
		ph_data_det_sky_top = sky_fl*tr_con_fit*exp_time*area*th*res_el
		#ph_data_det_sky_top = sky_fl*exp_time*area*th*res_el
		ph_data_det_sky_top[ph_data_det_sky_top < 1] = 1
		sky_noise = ph_data_det_sky_top**0.5
		rod_noise = 5**2
		sn_met = ph_rec_det/np.sqrt(ph_noise**2 + sky_noise**2+rod_noise)
	
	return ph_rec_det, ph_noise, sn_met, wav, mask_1+mask_2+mask_3


def doppler(wavlen, shift): ##shift in kms-1, +shift is away from each other and -ve is towards
	beta = (shift/2.99792458e5)
	factor = np.sqrt((1+beta)/(1-beta))
	wavlen_shift = wavlen*factor
	return wavlen_shift


def cross_corr_logl(f1, f2, mask): ##2 1D arrays of fluxes probably of same size, cross-corr and logl functions, mask is a boolean array for data array
	#n = len(f1) #length of arrays
	n_f = len(mask[mask == False]) ##count unmasked entries or False values
	if n_f == 0:
		return 0.0, 0.0  # all pixels masked — no information
	#n_f = np.count_nonzero(f1) # only non-zero elements
	#n_g = np.count_nonzero(f2) # only non-zero elements
	I_d = np.ones(n_f) #array of ones
	#m1 = np.mean(f1)
	#m1 = np.dot(f1,I_d)/n #mean of elements for f1
	a = f1[mask == False] #set masked entries to 0 so they don't contribute to calc
	b = f2[mask == False] #set corresponding model values to 0 as well
	m1 = np.dot(a,I_d)/n_f #mean of non-zero elements for f1
	#m2 = np.mean(f2)
	#m2 = np.dot(f2,I_d)/n #mean of non-zero elements for f2
	m2 = np.dot(b,I_d)/n_f
	d1 = a - m1 
	#d1[mask == True] = 0 #reset masked pixels to 0 as they would otherwise be artifically shifted to a -ve value
	d2 = b - m2
	#d2[mask == True] = 0 #same as d1 but for model
	#numerator = np.dot(d1,d2)/n
	numerator = np.dot(d1,d2)/n_f #only non-masked elements should contribute
	#d3 = np.dot(d1,d1)/n #sf^2
	d3 = np.dot(d1,d1)/n_f #sf^2, only non-masked elements
	#d4 = np.dot(d2,d2)/n #sg^2
	d4 = np.dot(d2,d2)/n_f #sg^2
	denominator = np.sqrt(d3*d4) #sfsg
	crss_crr = numerator/denominator
	#logl = -(n/2)*np.log(d3+d4-2*numerator)
	logl = -(n_f/2)*np.log(d3+d4-2*numerator) #using only unmasked elements
	return crss_crr, logl

def download_phoenix(teff, logg, feh, output_dir="."): #this function downloads the newest function of a phoenix spectrum directly from the source
	os.makedirs(output_dir, exist_ok=True)
	
	BASE_URL = "http://phoenix.astro.physik.uni-goettingen.de/data/HiResFITS" ###I queried Claude to find the https thing, it seems to have got it from the Starfish documentation
	
	feh_dir  = f"Z{feh:+.1f}"
	filename = (f"lte{int(teff):05d}-{logg:.2f}{feh:+.1f}"f".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits")
	spec_url   = f"{BASE_URL}/PHOENIX-ACES-AGSS-COND-2011/{feh_dir}/{filename}"
	spec_local = os.path.join(output_dir, filename)
	
	def progress(block_num, block_size, total_size):
		downloaded = block_num * block_size
		if total_size > 0:
			print(f"\r   {downloaded/1e6:.1f} MB / {total_size/1e6:.1f} MB ({min(downloaded/total_size*100, 100):.1f}%)", end="")
	
	if not os.path.exists(spec_local):
		print(f"Downloading spectrum: {filename}")
		urllib.request.urlretrieve(spec_url, spec_local, reporthook=progress)
		print(f"\n Spectrum saved to: {spec_local}")
	else:
		print("  Flux grid already exists, skipping.")
	
	# Wavelength grid
	wave_url   = f"{BASE_URL}/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
	wave_local = os.path.join(output_dir, "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
	if not os.path.exists(wave_local):
		print(f"  Downloading wavelength grid...")
		urllib.request.urlretrieve(wave_url, wave_local, reporthook=progress)
		print(f"\n Wavelength grid saved to: {wave_local}")
	else:
		print("  Wavelength grid already exists, skipping.")
	
	return spec_local, wave_local

def modelinjectdatacube(RVs_t, n_orders, n_spectra, n_pixels, flux_cube, wavlen_cube, fn_spl, transit): #RVs only for transit, transit is condition for transiting exoplanet
	flux_cube_injected = np.zeros((n_orders, n_spectra, n_pixels)) #final output cube
	flux_cube_transit = flux_cube[:,transit,:] #data with all orders and pixels but only transit spectra
	wavlen_cube_transit = wavlen_cube[:,transit,:] #same as abve but for wavelengths
	flux_cube_mid = np.zeros(flux_cube_transit.shape) #middle tier cube to store new values
	i = 0
	k = 0
	while k < len(RVs_t):
		w_out = doppler(wavlen_cube_transit[:,k,:], -RVs_t[k]) #RVs now flipped as we are doppler shifting in the opposite direction
		flux_new = splev(w_out, fn_spl, der = 0)
		while i < n_orders:
			flux_cube_mid[i,k,:] = flux_cube_transit[i,k,:]*flux_new[i] #for emission
			i = i+1
		i = 0
		k = k+1
	
	flux_cube_injected[:,~transit,:] = flux_cube[:,~transit,:] #retain the same values for non transit spectra
	flux_cube_injected[:,transit,:] = flux_cube_mid #change values for transit spectra
	del flux_cube_transit
	del flux_cube_mid
	
	return flux_cube_injected

def snr_3d():
	###simulate spectra according to location site an dtime###

	###spectrograph used and properties###
	#inst = 'crires+'
	inst = 'andes_ccd_carmenes' #instrument used
	modes = ['YJH']
	qual = 'good' #'good', 'avg', 'bad' #quality of night for telluric model
	t_exp = 60 #integration/exposure time
	
	###model###
	#dil_fac = '066'
	#dil_fac = '100'
	#model = '6' #fugacity regime
	
	#night = '1'
	
	night = '2'
	
	##observation mode
	ob = 'tm'

	#savedir = './andes_trials/exp60good/v1298taub/' #output directory to save files
	savedir = './andes_trials/exp60good_night2/v1298taub/cloudy/' #output directory to save files

	#if inst == 'crires+':
	#modes = ['y1028', 'y1029', 'j1226', 'j1228', 'j1232', 'h1559', 'h1567', 'k2148', 'k2166', 'k2192', 'k2217']
	#modes = ['j1228']
	#modes = [] #f0r rest of the instruments i.e. for non crires+ inst

	####for night###
	if inst == 'carmenes':
		t_duty = 34 ##for carmenes
		t_extra = 0
		res = 80400
		site = 'caha'
	
	if inst == 'giano':
		t_duty = 60 ##for giano
		t_extra = 0
		res = 50000
		site = 'Roque de los Muchachos'
	
	if inst == 'spirou':
		t_duty = 29 ##for spirou
		t_extra = 0
		res = 70000
		site = 'Canada-France-Hawaii Telescope'
	
	if inst == 'crires+':
		ndit = 1
		nod_seq = 'ABBA'
		t_duty = 2.4 + 1.43*(ndit-1) ##for crires+
		t_nod = 24
		res = 100000
		site = 'Cerro Paranal'
	
	if inst == 'andes':
		ndit = 1
		nod_seq = 'ABBA'
		t_duty = 2.4 + 1.43*(ndit-1) ##for crires+
		t_nod = 24
		res = 100000
		site = 'Cerro Armazones Observatory'
	
	if inst == 'andes_ccd_carmenes':
		t_duty = 34 ##for carmenes
		t_extra = 0
		res = 100000
		site = 'Cerro Armazones Observatory'
	
	fixed_target = FixedTarget.from_name('V1298tau')
	#site = 'caha' ###or 'Canada-France-Hawaii Telescope' - spirou, 'caha' - carmenes, 'Cerro Paranal' - crires+, 'Roque de los Muchachos' - giano
	#site = 'Roque de los Muchachos'
	#site = 'Canada-France-Hawaii Telescope'
	#site = 'Cerro Paranal'
	#site = 'Cerro Armazones Observatory'

	#observer = Observer.at_site('Canada-France-Hawaii Telescope')
	observer = Observer.at_site(site)
	
	####new#####
	if night == '1':
		#time = '2024-02-02 01:00:00' #for elt, night1
		time = '2030-01-21 01:30:00'
		obs_time = Time(time) ##for caha
		hrs_bf = -0.5
		hrs_af = 2
	
	if night == '2':
		time = '2030-12-25 02:00:00' #for elt, night2
		obs_time = Time(time) ##
		hrs_bf = -1 ##night 2
		hrs_af = 3 ###night 2
	
	if night == '3':
		time = '2032-12-18 01:30:00' #for elt, night2
		obs_time = Time(time) ##
		hrs_bf = -0.5 ##night 3
		hrs_af = 4 ###night 3
	
	if (inst == 'crires+' or inst == 'andes'):
		n_spectra = 4*int(((hrs_af-hrs_bf)*60*60)/(4*t_exp+4*t_duty+2*t_nod))
		print(n_spectra)
	else:
		n_spectra = int(((hrs_af-hrs_bf)*60*60 + t_duty)/(t_exp+t_duty))
		print(n_spectra)
	
	div = np.linspace(hrs_bf,hrs_af, n_spectra)

	observe_time = obs_time + div*u.hour

	alti = np.array(observer.altaz(observe_time, fixed_target).alt)
	airm = np.array(observer.altaz(observe_time, fixed_target).secz)
	
	plot_altitude(fixed_target, observer, observe_time, brightness_shading = True, airmass_yaxis = True)

	plt.savefig(savedir+'altair'+inst+'.png', bbox_inches = 'tight', dpi = 500)
	plt.show(block=False)
	plt.pause(2)
	plt.clf()
	
	### for calculating orbital phases
	
	fixed_target = SkyCoord.from_name('V1298tau')
	loc = EarthLocation.of_site(site)

	obs_time = Time(time, location = loc) ##for location
	
	'''
	if night == '1':
		div = np.linspace(hrs_bf,hrs_af, n_spectra) ###night 1
	if night == '2':
		div = np.linspace(hrs_bf,hrs_af, n_spectra) ##night 2
	'''
	div = np.linspace(hrs_bf,hrs_af, n_spectra)
	
	observe_time = obs_time + div*u.hour

	lt_bary = observe_time.light_travel_time(fixed_target)

	observe_time_bjd = observe_time.tdb.jd + lt_bary.jd  ##inspired by https://groups.google.com/g/astropy-dev/c/NBuhqUppCsA
	
	#mean_transit_BJD = 2457063.2096 ###for 55 cnc e, Bourrier et al 2018
	mean_transit_BJD = 2457067.0488 #v1298tau b David 2019
	#P = 0.7365474 ##in days, Bourrier et al 2021
	P = 24.1396 #in days, David 2019

	abs_ph = (observe_time_bjd - mean_transit_BJD)/P
	ph = abs_ph%1
	
	if ob == 'tm':
		ph[ph > 0.5] = ph[ph > 0.5] - 1
	
	plt.plot(ph, alti)
	plt.scatter(ph, alti, s = 8)
	plt.xlabel(r'Orbital Phase $\phi$')
	plt.ylabel('Altitude (degrees)')
	plt.title('Night '+night+' observation')
	plt.savefig(savedir+'altphase'+inst+'.png', bbox_inches = 'tight', dpi = 500)
	plt.show(block=False)
	plt.pause(2)
	plt.clf()
	

	plt.plot(ph, airm)
	plt.scatter(ph, airm, s = 8)
	plt.xlabel(r'Orbital Phase $\phi$')
	plt.ylabel('Airmass')
	plt.title('Night '+night+' observation')
	plt.savefig(savedir+'airmphase'+inst+'.png', bbox_inches = 'tight', dpi = 500)
	plt.show(block=False)
	plt.pause(2)
	plt.clf()
	

	barycorr = fixed_target.radial_velocity_correction(obstime = observe_time)  

	barycorr_kms = np.array(barycorr.to(u.km/u.s))

	d = 108.199 ##in pc, distance to star, from NASA exoplanet archive
	R_st = 1.43 ##in stellar radius

	vsys = 16.15 #kms-1, David 2019, gamma in nasa exoplanet archive
	e = 0.2 #<0.29, so used arbitrary
	omega = 85*0.0174533 ##references vary from 2+46-50 (rosenthal 2021) to 86+31-33 (bourrier 2018) with large error bars
	i = 89*0.0174533 #in radian
	a = 27*1.345*696340 ##in km
	P_s = P*24*60*60 ##in seconds
	tr_duration = 6.42 # transit duration in hours
	Kp = ((2*np.pi*a)*np.sin(i))/(P_s*(1-e**2)**0.5)
	
	RVs_t = ecc_RV(e, omega, ph, vsys, Kp, barycorr_kms)

	np.save(savedir+'bjd.npy', observe_time_bjd)
	np.save(savedir+'bar_rad_vel.npy', barycorr_kms)

	####now simulate the spectra for star, telluric absorption and doppler shifted planet according to RVs calculated

	###stellar data###
	stellar_spec = fits.open('lte05000-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits') #for V1298 Tau
	wavlen = fits.open('WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')

	spec_data = stellar_spec[0].data #flux density, in erg/s/cm**2/cm
	spec_data_a = spec_data*1e-8 #flux density, in erg/s/cm**2/A
	wavlen_data = wavlen[0].data #in angstrom

	wavlen_data_g = wavlen_data[((wavlen_data >= 4500) & (wavlen_data <= 17980))] ##restrict to only this wavelength range###
	spec_data_g = spec_data_a[((wavlen_data >= 4500) & (wavlen_data <= 17980))]

	wavlen_data_rg, spec_data_c, fn_spl_st = broaden(wavlen_data_g, spec_data_g, res, splinefit = True)

	###Flux density at top of earth's atmosphere###

	star_flux_ph = photonfluxearthtop(wavlen_data_rg, spec_data_c, d, R_st)
	fn_star_ph = splinefit(wavlen_data_rg, star_flux_ph)
	
	###generate list of ppv values, avoid random fluctations by sorting the list###
	np.random.seed(1) ##let the ppv variation per night have the same randomness
	
	if inst == 'andes_ccd_carmenes':
		if qual == 'good':
			#p_list = np.random.choice([2.5, 3.5], n_spectra)
			#p_list = np.random.choice([2.5], n_spectra)
			p_list = np.random.choice([3.5], n_spectra)
			p_list.sort()
		if qual == 'avg':
			p_list = np.random.choice([5.0, 7.5], n_spectra)
			p_list.sort()
		if qual == 'bad':
			p_list = np.random.choice([10.0, 20.0], n_spectra)
			p_list.sort()	
	i = 0
	
	while i < n_spectra:
		
		ppv = p_list[i]
		
		tm_fits = fits.open('./atm_transmission/skytable_a_1.0_p_'+str(ppv)+'.fits') 
		data = tm_fits[1].data
		w_mod = data['lam']*10 #from nm to angstrom, ##already obtained at the relevant wavelengths of 0.35 to 2.5 microns, otherwise trim
		tr_mod = data['trans'] #transmission
		tm_fits.close()
		
		with np.errstate(divide='ignore', invalid='ignore'):
			tr_mod = np.exp(airm[i]*np.log(tr_mod)) #effect of airmass variance with time
		wav_trreg, tr_con, fn_tr_con = broaden(w_mod, tr_mod, res, splinefit = True)
		
		###sky emission###
		sky_flux = data['flux']*1e-8 ##in ph/s/cm**2/A
		wav_skyflux_reg, sky_flux_con, fn_skyflux_con = broaden(w_mod, sky_flux, res, splinefit = True)	
		
		###calculate signal to noise depending on instrument###
		
		#ph_det_top_comb, noise_comb, sn_met_comb, wav, mask_comb = signalcalc(inst, modes, t_exp, fn_pl_star_ph, fn_tr_con)
		ph_det_top_st, noise_st, sn_met_st, wav_st, mask_st = signalcalc(inst, modes, t_exp, fn_star_ph, fn_tr_con, fn_skyflux_con) #only star
		
		if night == '1':
			np.random.seed(1*(i+1)) #exposure and night dependant for night 1, ask?
		if night == '2':
			np.random.seed(100*(i+1)) #exposure and night dependant for night 2, ask?
		if night == '3':
			np.random.seed(10*(i+1)) #exposure and night dependant for night 3, ask?
		
		var = np.random.uniform(low=0.95, high=1.05) ##modulation in flux in each expsore due to miscellaneous sources
		
		#ph_det_top_comb = ph_det_top_comb*var
		ph_det_top_st = ph_det_top_st*var
		
		if inst != 'andes':
			n_orders, n_pixels = ph_det_top_st.shape
		if inst == 'andes':
			n_pixels = len(ph_det_top_st)
		
		#damp = 1 ##probably n_spectra**0.5?
		#np.random.seed(1*(i+1)) ##night and exposure dependant
		
		#wh_noise_mat = (1.0/damp)*np.random.normal(np.zeros_like(ph_det_top_st), noise_st) ##add actor to dampen
		
		#ph_det_top_comb_noisy = ph_det_top_comb+wh_noise_mat
		#ph_det_top_st_noisy = ph_det_top_st+wh_noise_mat
		
		if i == 0:
			if inst != 'andes':
				flux_startell = np.zeros((n_spectra, n_orders, n_pixels))
				#flux_startell_noisy = np.zeros((n_spectra, n_orders, n_pixels))
				#flux_err = np.zeros((n_spectra, n_orders, n_pixels))
				wavlen = np.zeros((n_spectra, n_orders, n_pixels))
				#mask_arr = np.zeros((n_spectra, n_orders, n_pixels))
				snr_arr = np.zeros((n_spectra, n_orders, n_pixels))
			if inst == 'andes':
				flux_startell = np.zeros((n_spectra, n_pixels))
				#flux_startell_noisy = np.zeros((n_spectra, n_orders, n_pixels))
				#flux_err = np.zeros((n_spectra, n_orders, n_pixels))
				wavlen = np.zeros((n_spectra, n_pixels))
				#mask_arr = np.zeros((n_spectra, n_orders, n_pixels))
				snr_arr = np.zeros((n_spectra, n_pixels))
		
		#mask_add = ph_det_top_st_noisy < 1
		#ph_det_top_st_noisy[ph_det_top_st_noisy < 1] = 1 #around 1681 values, add sn mask here and check
			
		
		flux_startell[i] = ph_det_top_st
		#flux_startell_noisy[i] = ph_det_top_st_noisy
		#flux_err[i] = wh_noise_mat
		wavlen[i] = wav_st
		#mask_arr[i] = mask_st+mask_add
		snr_arr[i] = sn_met_st
		i = i+1
	
	if inst != 'andes':
		flux_cube_startell = np.swapaxes(flux_startell,0,1)
		#flux_cube_startell_noisy = np.swapaxes(flux_startell_noisy,0,1)
		#flux_err_cube = np.swapaxes(flux_err,0,1)
		wavlen_cube = np.swapaxes(wavlen,0,1)
		#mask_cube = np.swapaxes(mask_arr,0,1)
		snr_cube = np.swapaxes(snr_arr,0,1)
	if inst == 'andes': 
		flux_cube_startell = np.array([flux_startell])
		#flux_cube_startell_noisy = np.swapaxes(flux_startell_noisy,0,1)
		#flux_err_cube = np.swapaxes(flux_err,0,1)
		wavlen_cube = np.array([wavlen])
		#mask_cube = np.swapaxes(mask_arr,0,1)
		snr_cube = np.array([snr_arr])
	
	### now calculate the star+planet flux### at detector for each spectra and construct cube
	
	##for V1298 Tau##
	pl_data = pd.read_table('./models/Cloud_Updated_V1298_Tau_b/Ratri_V1298_Tau_b_ANDES_H2O.txt', delimiter = '\t+', header = 0, names = ['wav', 'tr_d', 'tr_d_no', 'tr_d_only'], engine = 'python')
	p_flux = pl_data['tr_d'].to_numpy()
	wav_p = pl_data['wav'].to_numpy()
	wav_p = wav_p*1e4 ##from micron to angstrom

	wavlen_p = wav_p[((wav_p >= 4000) & (wav_p <= 18300))]
	p_srflux = p_flux[((wav_p >= 4000) & (wav_p <= 18300))]
	wreg, psr_con = broaden(wavlen_p, p_srflux, res, splinefit = False)
	
	a = 1 ##scale factor for injection
	
	if ob == 'em':
		psr_flip_con = 1 + a*(psr_con)
		psr_flip_con_c = psr_flip_con
		
		egress = ph[-1] ##dayside obs
		ingress = ph[0] ##dayside obs
		transit = ((ph >= ingress) & (ph <= egress)) #selects indices for phases between the ingress and egress phases
	
	if ob == 'tm':
		psr_flip_con = 1 + a*np.negative(psr_con) #broadened fluxes at regridded wavelengths
		#fmod_flip_con_c = fmod_flip_con/(1-tr_depth) ##normalize transit depth
		#psr_flip_con_c = psr_flip_con/np.max(psr_flip_con) ##normalize transit depth by its maximum
		psr_flip_con_ser = pd.Series(psr_flip_con)
		psr_flip_rollmax = psr_flip_con_ser.rolling(500,1).max() ##fimd rolling minimum of 5 values, with a minimum window of 1
		psr_flip_con_c = psr_flip_con_ser/psr_flip_rollmax ##normalize transit depth by its rolling maximum (to better capture the continuum)
		
		egress = 0.5*(tr_duration/(P*24))
		ingress = -egress
		transit = ((ph >= ingress) & (ph <= egress)) #selects indices for phases between the ingress and egress phases
	
	f_p = splinefit(wreg, psr_flip_con_c)
	
	n_orders, n_spectra, n_pixels = flux_cube_startell.shape
	
	RVs = RVs_t[transit]
	flux_cube_st_pl = modelinjectdatacube(RVs, n_orders, n_spectra, n_pixels, flux_cube_startell, wavlen_cube, f_p, transit)
	noise_st_pl = flux_cube_st_pl**0.5
	
	damp = 1
	if night == '1':
		np.random.seed(3) #for night 1
	if night == '2':
		np.random.seed(100) #for night 2
	if night == '3':
		np.random.seed(10) #for night 3
	
	flux_err_cube = (1.0/damp)*np.random.normal(np.zeros_like(flux_cube_st_pl), noise_st_pl) ##white noise
	
	flux_cube = flux_cube_st_pl + flux_err_cube
	
	
	np.save(savedir+'flux_cube_'+inst+'.npy', flux_cube)
	np.save(savedir+'flux_cube_startell'+inst+'.npy', flux_cube_startell)
	np.save(savedir+'flux_err_cube_'+inst+'.npy', flux_err_cube)
	np.save(savedir+'wavlen_cube_'+inst+'.npy', wavlen_cube)
	np.save(savedir+'snr_cube_'+inst+'.npy', snr_cube)
	
	plt.imshow(flux_cube[0], aspect = 'auto', origin = 'lower')
	plt.show(block=False)
	plt.pause(2)
	plt.clf()



if __name__ == "__main__":
	snr_3d()



# ===========================================================================
# ratri() — automated night selection and observation simulator
#
# Replaces the manual night/time setup in snr_3d() with automated selection
# using phase_visibility logic, then simulates the chosen night.
#
# Structure:
#   ratri()                  — top-level entry point
#   _select_night()          — find best observable night in a date range
#   _build_time_grid()       — construct exposure time grid for chosen night
#   _compute_orbit()         — BJDs, orbital phases, barycentric RVs
#   _load_stellar_spectrum() — PHOENIX loading, broadening, photon flux
#   _build_flux_cube()       — telluric + signal loop over exposures
#   _inject_planet()         — Doppler shift and inject planet model
#   _add_noise_and_save()    — white noise draw, save cubes, diagnostic plots
# ===========================================================================

from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from astroplan import Observer, FixedTarget
import astropy.units as u
import numpy as np


# ---------------------------------------------------------------------------
# Observatory definitions (mirrors phase_visibility.py)
# ---------------------------------------------------------------------------

# PWV quality tiers: (pwv_min, pwv_median, pwv_max) in mm
# Used to parameterise a lognormal distribution for each exposure.
# References:
#   Kerber et al. 2012 (Paranal, median ~2.4 mm)
#   Garcia-Lorenzo et al. 2010 (ORM/La Palma, median ~3.8 mm)
#   Otarola et al. 2010 (Mauna Kea, median ~2 mm)
#   Sanchez et al. 2007 (Calar Alto, wetter than Paranal/Mauna Kea)
_PWV_TIERS = {
	'excellent': (0.1,  0.3,  0.5),
	'very_good': (1.0,  1.25, 1.5),
	'good':      (2.5,  3.0,  3.5),
	'avg':       (5.0,  6.25, 7.5),
	'bad':       (10.0, 15.0, 20.0),
}

_OBSERVATORIES = {
	'crires+':           dict(site_name='Cerro Paranal',          lon=-70.4045,  lat=-24.6268, height=2648.0, default_qual='good'),
	'carmenes':          dict(site_name='Calar Alto',             lon=-2.5467,   lat=37.2236,  height=2168.0, default_qual='avg'),
	'giano':             dict(site_name='Roque de los Muchachos', lon=-17.8900,  lat=28.7540,  height=2387.0, default_qual='good'),
	'spirou':            dict(site_name='Mauna Kea',              lon=-155.4683, lat=19.8255,  height=4204.0, default_qual='very_good'),
	'andes':             dict(site_name='Cerro Armazones',        lon=-70.1918,  lat=-24.5899, height=3046.0, default_qual='good'),
	'andes_ccd_carmenes':dict(site_name='Cerro Armazones',        lon=-70.1918,  lat=-24.5899, height=3046.0, default_qual='good'),
	'tmt_ccd_giano':     dict(site_name='Mauna Kea',              lon=-155.4683, lat=19.8255,  height=4204.0, default_qual='very_good'),
	# National Large Optical-infrared Telescope (NLOT), Hanle, Leh, India
	# Site: lon=78°57'58"E, lat=32°46'46"N, alt=4535m
	# Median PWV: 1-2 mm (very_good tier); ~250 spectroscopic nights/year
	# CCD configuration: GIANO (broadest NIR coverage, 0.95-2.45 um)
	# First-light instruments still being finalised as of 2025
	'nlot_ccd_giano':    dict(site_name='Hanle',                  lon=78.9661,   lat=32.7794,  height=4535.0, default_qual='very_good'),
}

_INSTRUMENT_PARAMS = {
	'crires+':          dict(res=100000, t_duty=3.83, t_nod=24, nod_seq='ABBA'),
	'carmenes':         dict(res=80400,  t_duty=34,   t_nod=0,  nod_seq=None),
	'giano':            dict(res=50000,  t_duty=60,   t_nod=0,  nod_seq=None),
	'spirou':           dict(res=70000,  t_duty=29,   t_nod=0,  nod_seq=None),
	# andes modes: 'YJ', 'H', 'K', 'UBV', 'RIZ' (pass as list, e.g. modes=['YJ','K'])
	'andes':            dict(res=100000, t_duty=3.83, t_nod=24, nod_seq='ABBA'),
	# andes_ccd_carmenes modes: 'YJH' only
	'andes_ccd_carmenes':dict(res=100000,t_duty=34,   t_nod=0,  nod_seq=None),
	# NLOT uses GIANO CCD config — resolution and duty cycle TBC pending first-light instrument selection
	'nlot_ccd_giano':   dict(res=60000,  t_duty=60,   t_nod=0,  nod_seq=None),  # res TBC, lower end estimate
}


# ---------------------------------------------------------------------------
# Sub-function 1: Night selection
# ---------------------------------------------------------------------------



def _longest_contiguous_block(mask, dt_hours):
	"""
	Given a boolean mask and a time step size in hours, return:
	  longest_hours : duration of the longest contiguous True block
	  total_hours   : total time across all True blocks
	  start_idx     : index into the original mask of the block start
	  end_idx       : index into the original mask of the block end (exclusive)
	"""
	indices = np.where(mask)[0]
	if len(indices) == 0:
		return 0.0, 0.0, 0, 0

	total_hours = len(indices) * dt_hours

	best_len, best_start = 1, 0
	cur_len,  cur_start  = 1, 0

	for i in range(1, len(indices)):
		if indices[i] == indices[i - 1] + 1:
			cur_len += 1
		else:
			if cur_len > best_len:
				best_len, best_start = cur_len, cur_start
			cur_start = i
			cur_len   = 1

	if cur_len > best_len:
		best_len, best_start = cur_len, cur_start

	start_idx = indices[best_start]
	end_idx   = indices[best_start + best_len - 1] + 1  # exclusive

	return best_len * dt_hours, total_hours, start_idx, end_idx

def _spinner(message, stop_event):
	"""Animate a spinner in the terminal while stop_event is not set."""
	import sys, time, itertools
	for ch in itertools.cycle('|/-\\'):
		if stop_event.is_set():
			break
		sys.stdout.write(f'\r  {message} {ch}')
		sys.stdout.flush()
		time.sleep(0.1)
	sys.stdout.write(f'\r  {message} done.\n')
	sys.stdout.flush()


def _select_night(target_name, inst, t_start, t_end,
                  T0, period, phase_min, phase_max,
                  airmass_min=1.0, airmass_max=2.0,
                  sun_horizon=-18.0, dt_min=10,
                  min_duration=1.0):
	"""
	Search a date range for nights where the target is observable within
	the requested orbital phase window, and return a ranked summary.

	Parameters
	----------
	target_name  : str   Simbad-resolvable target name
	inst         : str   instrument key from _OBSERVATORIES
	t_start      : astropy.time.Time
	t_end        : astropy.time.Time
	T0           : astropy.time.Time   mid-transit epoch (BJD_TDB)
	period       : astropy.units.Quantity   orbital period
	phase_min    : float   phase range lower bound in [-0.5, 0.5)
	phase_max    : float   phase range upper bound; if > phase_min wraps through +-0.5
	airmass_min  : float   (default 1.0)
	airmass_max  : float   (default 2.0)
	sun_horizon  : float   degrees (default -18, astronomical twilight)
	dt_min       : float   time grid step in minutes (default 10)
	min_duration : float   minimum hours in phase window per night (default 1.0)

	Returns
	-------
	list of dicts, one per qualifying night, sorted by total time in window
	(descending). Each dict has keys:
	  night        : str   e.g. '2026-04-18/19'
	  evening_date : str   e.g. '2026-04-18'  (use to set obs_time)
	  total_hours  : float
	  phase_range  : (float, float)   actual phase coverage that night
	  airmass_range: (float, float)
	  times        : astropy.time.Time
	  phases       : numpy.ndarray
	  airmass      : numpy.ndarray
	"""
	obs_params = _OBSERVATORIES[inst]
	location   = EarthLocation.from_geodetic(
		lon    = obs_params['lon'] * u.deg,
		lat    = obs_params['lat'] * u.deg,
		height = obs_params['height'] * u.m,
	)
	observer = Observer(location=location, name=obs_params['site_name'], timezone='UTC')

	print(f'Resolving coordinates for "{target_name}"...')
	coord  = SkyCoord.from_name(target_name)
	target = FixedTarget(coord, name=target_name)
	print(f'  RA={coord.ra.deg:.6f} deg  Dec={coord.dec.deg:.6f} deg')

	# Warn user if search range is far enough ahead that ERFA/IERS
	# precision will degrade, then suppress the resulting ERFA warnings
	import warnings
	from astropy.utils.exceptions import AstropyWarning
	import erfa
	years_ahead = (t_end.jd - 2451545.0) / 365.25  # years from J2000
	if years_ahead > 20:
		print(
			f'  Note: search end date is ~{years_ahead:.0f} years from J2000. '
			f'ERFA/IERS Earth-orientation data will not be available that far ahead. '
			f'Results are representative but astrometric precision will be reduced.'
		)

	# Build time grid and compute observability — show spinner while running
	import threading
	dt      = dt_min * u.minute
	n_steps = int((t_end - t_start) / dt)
	search_msg = (f'Searching {t_start.iso[:10]} to {t_end.iso[:10]} '
	              f'({n_steps} steps of {dt_min} min)')
	stop_event = threading.Event()
	spin_thread = threading.Thread(target=_spinner, args=(search_msg, stop_event), daemon=True)
	spin_thread.start()

	try:
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', erfa.ErfaWarning)
			warnings.simplefilter('ignore', AstropyWarning)
			times   = t_start + np.arange(n_steps) * dt

		# Orbital phases in [-0.5, 0.5)
		phases = ((times.tdb.jd - T0.jd) / period.to(u.day).value + 0.5) % 1.0 - 0.5

		# Airmass and sun altitude — suppress ERFA warnings for future dates
		from astropy.coordinates import AltAz
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', erfa.ErfaWarning)
			warnings.simplefilter('ignore', AstropyWarning)
			altaz_frame = AltAz(obstime=times, location=location)
			altaz       = coord.transform_to(altaz_frame)
			alt_deg     = altaz.alt.deg
			sun_alt     = observer.sun_altaz(times).alt.deg

		with np.errstate(invalid='ignore', divide='ignore'):
			airmass = np.where(alt_deg > 0, 1.0 / np.sin(np.radians(alt_deg)), np.inf)

		# Observability mask
		sun_down   = np.array(sun_alt) < sun_horizon
		in_airmass = (airmass >= airmass_min) & (airmass <= airmass_max)
		observable = sun_down & in_airmass
	finally:
		stop_event.set()
		spin_thread.join()

	# Phase window mask
	if phase_min <= phase_max:
		in_range = (phases >= phase_min) & (phases <= phase_max)
	else:
		in_range = (phases >= phase_min) | (phases <= phase_max)

	mask = observable & in_range

	# Assign nights using UTC noon boundary
	utc_offset  = obs_params['lon'] / 15.0   # hours, 360 degrees/24 hours
	shifted_jd  = times.utc.jd - (12.0 - utc_offset) / 24.0
	evening_jds = np.floor(shifted_jd)
	# Vectorised night label construction — single Time array call instead
	# of one Time object per step, which is much faster over long date ranges
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', erfa.ErfaWarning)
		warnings.simplefilter('ignore', AstropyWarning)
		evening_dts = Time(evening_jds,     format='jd', scale='utc').to_datetime()
		morning_dts = Time(evening_jds + 1, format='jd', scale='utc').to_datetime()
	evening_strs = np.array([d.strftime('%Y-%m-%d') for d in evening_dts])
	morning_strs = np.array([d.strftime('%d')       for d in morning_dts])
	night_labels = np.array([f'{e}/{m}' for e, m in zip(evening_strs, morning_strs)])

	matched_nights = night_labels[mask]
	dt_hours       = dt.to(u.hour).value

	# Spinner for the cataloguing step
	unique_nights = sorted(set(matched_nights))
	stop_event2   = threading.Event()
	spin_thread2  = threading.Thread(
		target = _spinner,
		args   = (f'Cataloguing {len(unique_nights)} candidate nights', stop_event2),
		daemon = True,
	)
	spin_thread2.start()

	qualifying = []
	try:
		for night in unique_nights:
			sel = (night_labels == night) & mask
			longest_hours, total_hours, blk_start, blk_end = _longest_contiguous_block(sel, dt_hours)
			if longest_hours < min_duration:
				continue

			ph_sel  = phases[blk_start:blk_end]
			am_sel  = airmass[blk_start:blk_end]
			t_sel   = times[blk_start:blk_end]
			ev_date = night.split('/')[0]

			if len(ph_sel) == 0:
				continue

			# Build phase string
			wraps = phase_min > phase_max
			if wraps:
				high_phases = ph_sel[ph_sel >= phase_min]
				low_phases  = ph_sel[ph_sel <= phase_max]
				if len(high_phases) > 0 and len(low_phases) > 0:
					phase_str = (
						f'[{high_phases.min():+.4f}, {high_phases.max():+.4f}]'
						f' & [{low_phases.min():+.4f}, {low_phases.max():+.4f}]'
					)
				elif len(high_phases) > 0:
					phase_str = f'[{high_phases.min():+.4f}, {high_phases.max():+.4f}] (positive wing only)'
				elif len(low_phases) > 0:
					phase_str = f'[{low_phases.min():+.4f}, {low_phases.max():+.4f}] (negative wing only)'
				else:
					phase_str = f'[{ph_sel.min():+.4f}, {ph_sel.max():+.4f}]'
			else:
				phase_str = f'[{ph_sel.min():+.4f}, {ph_sel.max():+.4f}]'

			qualifying.append(dict(
				night           = night,
				evening_date    = ev_date,
				total_hours     = longest_hours,
				total_hours_all = total_hours,
				phase_str       = phase_str,
				phase_range     = (ph_sel.min(), ph_sel.max()),
				airmass_range   = (am_sel.min(), am_sel.max()),
				obs_start       = t_sel[0],
				obs_end         = t_sel[-1],
				times           = t_sel,
				phases          = ph_sel,
				airmass         = am_sel,
			))
	finally:
		stop_event2.set()
		spin_thread2.join()

	qualifying.sort(key=lambda x: x['total_hours'], reverse=True)

	print(f'\nFound {len(qualifying)} qualifying nights '
	      f'(>= {min_duration:.1f} h in phase window, airmass [{airmass_min}, {airmass_max}]):')
	print(f'  {"Rank":<5} {"Night":<16} {"Obs. Hours (Total)":<22} {"Phase Coverage":<52} {"Airmass"}')
	print('  ' + '-' * 109)
	for rank, q in enumerate(qualifying):
		hrs_str   = f'{q["total_hours"]:.1f} ({q["total_hours_all"]:.1f})'
		phase_str = q['phase_str'].ljust(50)
		print(
			f'  {rank:<5} '
			f'{q["night"]:<16} '
			f'{hrs_str:<22}'
			f'{phase_str:<52}'
			f'[{q["airmass_range"][0]:.2f}, {q["airmass_range"][1]:.2f}]'
		)

	return qualifying


# ---------------------------------------------------------------------------
# Sub-function 2: Build exposure time grid for chosen night
# ---------------------------------------------------------------------------

def _build_time_grid(inst, obs_start, obs_end, t_exp):
	"""
	Construct the actual exposure time grid for a chosen night.

	The start and end of the observable window come directly from _select_night.
	The central time of that window is used as the reference, and hrs_before /
	hrs_after are derived symmetrically around it — consistent with the original
	ratri approach but determined automatically from the night selection.

	Parameters
	----------
	inst      : str   instrument key
	obs_start : astropy.time.Time   start of observable window
	obs_end   : astropy.time.Time   end of observable window
	t_exp     : float exposure time in seconds

	Returns
	-------
	observe_time : astropy.time.Time array
	n_spectra    : int
	"""
	params  = _INSTRUMENT_PARAMS[inst]
	t_duty  = params['t_duty']
	t_nod   = params['t_nod']
	nod_seq = params['nod_seq']

	window_seconds = (obs_end - obs_start).to(u.s).value
	half_window    = window_seconds / 2.0
	obs_mid        = obs_start + half_window * u.s
	hrs_before     = -half_window / 3600.0
	hrs_after      =  half_window / 3600.0

	if nod_seq == 'ABBA':
		n_spectra = 4 * int(window_seconds / (4 * t_exp + 4 * t_duty + 2 * t_nod))
	else:
		n_spectra = int((window_seconds + t_duty) / (t_exp + t_duty))

	n_spectra    = max(n_spectra, 1)
	div          = np.linspace(hrs_before, hrs_after, n_spectra)
	observe_time = obs_mid + div * u.hour

	print(f'Instrument   : {inst}  (t_exp={t_exp}s, t_duty={t_duty}s)')
	print(f'Window       : {obs_start.iso} to {obs_end.iso}')
	print(f'Centre       : {obs_mid.iso}')
	print(f'hrs_before   : {hrs_before:.2f} h  hrs_after: {hrs_after:.2f} h')
	print(f'n_spectra    : {n_spectra}')

	return observe_time, n_spectra


# ---------------------------------------------------------------------------
# Sub-function 3: Orbital phases and barycentric RVs
# ---------------------------------------------------------------------------

def _compute_orbit(observe_time, inst, target_name,
                   T0, period, mean_transit_BJD,
                   e, omega, vsys, Kp, ob,
                   tr_duration_h=None):
	"""
	Compute BJDs, orbital phases, barycentric correction, and planet RVs.

	Parameters
	----------
	observe_time     : astropy.time.Time array
	inst             : str
	target_name      : str
	T0               : astropy.time.Time   mid-transit epoch
	period           : astropy.units.Quantity
	mean_transit_BJD : float   BJD_TDB of reference transit
	e                : float   eccentricity
	omega            : float   argument of periastron in radians
	vsys             : float   systemic velocity in km/s
	Kp               : float   planet RV semi-amplitude in km/s
	ob               : str     'tm' (transit) or 'em' (emission/occultation)

	Returns
	-------
	dict with keys: bjd, ph, barycorr_kms, RVs_t, transit_mask, airmass
	"""
	obs_params = _OBSERVATORIES[inst]
	location   = EarthLocation.from_geodetic(
		lon    = obs_params['lon'] * u.deg,
		lat    = obs_params['lat'] * u.deg,
		height = obs_params['height'] * u.m,
	)

	coord        = SkyCoord.from_name(target_name)
	fixed_target = FixedTarget(coord, name=target_name)

	# Light travel time to barycentre
	obs_time_loc = Time(observe_time.utc.jd, format='jd', scale='utc', location=location)
	lt_bary      = obs_time_loc.light_travel_time(coord)
	bjd          = obs_time_loc.tdb.jd + lt_bary.jd

	# Barycentric RV correction
	t_no_loc     = Time(observe_time.utc.jd, format="jd", scale="utc")
	barycorr     = coord.radial_velocity_correction(obstime=t_no_loc, location=location)
	barycorr_kms = np.array(barycorr.to(u.km / u.s))

	# Orbital phases in [-0.5, 0.5)
	period_days = period.to(u.day).value
	ph = ((bjd - mean_transit_BJD) / period_days + 0.5) % 1.0 - 0.5

	# Airmass
	observer = Observer(location=location, timezone='UTC')
	from astropy.coordinates import AltAz
	altaz_frame = AltAz(obstime=observe_time, location=location)
	altaz       = coord.transform_to(altaz_frame)
	alt_deg     = altaz.alt.deg
	with np.errstate(invalid='ignore', divide='ignore'):
		airm = np.where(alt_deg > 0, 1.0 / np.sin(np.radians(alt_deg)), np.inf)

	# Transit/emission window mask
	if ob == 'tm':
		# Use empirical transit duration from literature
		half_dur     = 0.5 * tr_duration_h / (period_days * 24.0)
		transit_mask = np.abs(ph) <= half_dur
	else:
		# Emission: planet is visible on dayside except during occultation
		# If planet transits (tr_duration_h > 0), compute occultation duration
		# from empirical transit duration via Winn 2010 eq. 34:
		#   T_occ/T_tra = (1 + e*sin(omega)) / (1 - e*sin(omega))
		# omega is already in radians internally.
		# For circular orbits the ratio = 1, so T_occ = T_tra.
		if tr_duration_h is not None and tr_duration_h > 0:
			occ_ratio    = (1 + e * np.sin(omega)) / (1 - e * np.sin(omega))
			occ_dur_h    = tr_duration_h * occ_ratio
			half_occ     = 0.5 * occ_dur_h / (period_days * 24.0)
			# Occultation centred at phase +-0.5; exclude those phases
			transit_mask = np.abs(np.abs(ph) - 0.5) > half_occ
			print(f'Occultation duration: {occ_dur_h:.3f} h '
			      f'(ratio {occ_ratio:.4f} x transit duration)')
		else:
			# No transit / duration unknown — no occultation to mask
			# (non-transiting planet or dayside-only observation)
			transit_mask = np.ones(len(ph), dtype=bool)

	# Planet RVs using eccentric orbit
	RVs_t = ecc_RV(e, omega, ph, vsys, Kp, barycorr_kms)

	return dict(
		bjd          = bjd,
		ph           = ph,
		barycorr_kms = barycorr_kms,
		RVs_t        = RVs_t,
		transit_mask = transit_mask,
		airmass      = airm,
	)


# ---------------------------------------------------------------------------
# Sub-function 4: Stellar spectrum
# ---------------------------------------------------------------------------

def _load_stellar_spectrum(phoenix_spec_path, phoenix_wave_path,
                           res, d_pc, R_st_rsun):
	"""
	Load and broaden a PHOENIX stellar spectrum, return photon flux spline.

	PHOENIX/ACES library: Husser et al. 2013, A&A 553, A6
	https://phoenix.astro.physik.uni-goettingen.de

	Parameters
	----------
	phoenix_spec_path : str   path to PHOENIX flux FITS file
	phoenix_wave_path : str   path to PHOENIX wavelength FITS file
	res               : float spectral resolution R
	d_pc              : float distance to star in parsec
	R_st_rsun         : float stellar radius in solar radii
	wav_min, wav_max  : float wavelength range in Angstrom

	Returns
	-------
	fn_star_ph : spline tuple   photon flux at top of atmosphere
	wavlen_rg  : numpy.ndarray  regridded wavelength array in Angstrom
	"""
	from astropy.io import fits

	stellar_spec = fits.open(phoenix_spec_path)
	wavlen_fits  = fits.open(phoenix_wave_path)

	spec_data   = stellar_spec[0].data * 1e-8   # erg/s/cm^2/A
	wavlen_data = wavlen_fits[0].data            # Angstrom

	stellar_spec.close()
	wavlen_fits.close()

	# Use full PHOENIX wavelength range — no hardcoded clip.
	# broaden/splinefit evaluates only at instrument wavelengths.
	wavlen_g     = wavlen_data
	spec_g       = spec_data

	wavlen_rg, spec_c, fn_spl_st = broaden(wavlen_g, spec_g, res, splinefit=True)
	star_flux_ph = photonfluxearthtop(wavlen_rg, spec_c, d_pc, R_st_rsun)
	fn_star_ph   = splinefit(wavlen_rg, star_flux_ph)

	return fn_star_ph, wavlen_rg


# ---------------------------------------------------------------------------
# Sub-function 5: Build flux cube (telluric + signal loop)
# ---------------------------------------------------------------------------


def _sample_pwv(qual, n_spectra, seed=1):
	"""
	Draw n_spectra PWV values (mm) from a lognormal distribution
	parameterised by the quality tier.

	The lognormal is parameterised so that its median equals the tier
	median and sigma is set so ~95% of draws fall within the tier bounds.
	PWV distributions at astronomical sites are well-described by lognormals:
	Kerber et al. 2012 (Paranal), Garcia-Lorenzo et al. 2010 (ORM),
	Otarola et al. 2010 (Mauna Kea), Sanchez et al. 2007 (Calar Alto).
	median and the sigma in log-space is chosen so ~95% of draws
	fall within (pwv_min, pwv_max). Draws are clipped to [pwv_min, pwv_max]
	and sorted so PWV varies smoothly across the night.

	Parameters
	----------
	qual      : str   one of 'excellent', 'very_good', 'good', 'avg', 'bad'
	n_spectra : int
	seed      : int   random seed

	Returns
	-------
	pwv_array : numpy.ndarray  shape (n_spectra,)
	"""
	if qual not in _PWV_TIERS:
		raise ValueError(
			f'Unknown PWV quality tier "{qual}". '
			f'Choose from: {list(_PWV_TIERS.keys())}'
		)
	pwv_min, pwv_med, pwv_max = _PWV_TIERS[qual]

	# Lognormal: ln(X) ~ N(mu, sigma)
	# median of lognormal = exp(mu) => mu = ln(pwv_med)
	# Set sigma so that exp(mu + 2*sigma) ~ pwv_max
	# => sigma = (ln(pwv_max) - ln(pwv_med)) / 2
	mu    = np.log(pwv_med)
	sigma = (np.log(pwv_max) - np.log(pwv_med)) / 2.0

	np.random.seed(seed)
	draws = np.random.lognormal(mean=mu, sigma=sigma, size=n_spectra)
	draws = np.clip(draws, pwv_min, pwv_max)
	draws.sort()
	return draws

def _build_flux_cube(inst, modes, t_exp, res, n_spectra, airm,
                     fn_star_ph, qual, atm_dir, seed_base=1):
	"""
	Loop over exposures, apply telluric absorption and sky emission,
	compute detected photon counts and SNR per exposure.

	Parameters
	----------
	inst        : str
	modes       : list of str
	t_exp       : float   exposure time in seconds
	res         : float   spectral resolution
	n_spectra   : int
	airm        : numpy.ndarray   airmass per exposure
	fn_star_ph  : spline tuple
	qual        : str   one of 'excellent', 'very_good', 'good', 'avg', 'bad'
	atm_dir     : str   directory containing skytable_a_1.0_p_*.fits files
	seed_base   : int   random seed base for flux modulation

	Returns
	-------
	flux_cube_startell : numpy.ndarray  (n_orders, n_spectra, n_pixels)
	wavlen_cube        : numpy.ndarray  (n_orders, n_spectra, n_pixels)
	snr_cube           : numpy.ndarray  (n_orders, n_spectra, n_pixels)
	pwv_array          : numpy.ndarray  (n_spectra,)  PWV per exposure in mm
	"""
	import glob
	from astropy.io import fits

	# Draw PWV values from lognormal distribution for this quality tier
	p_list = _sample_pwv(qual, n_spectra, seed=seed_base)
	print(f'  PWV: min={p_list.min():.2f} mm  median={np.median(p_list):.2f} mm  max={p_list.max():.2f} mm')

	# Build sorted PWV grid from available skytable files
	# Skytable files generated with ESO SkyCalc — Noll et al. 2012, A&A 543, A92
	# and Jones et al. 2013, A&A 560, A91
	# https://www.eso.org/observing/etc/bin/gen/form?INS.MODE=swspectr+INS.NAME=SKYCALC
	pwv_files = sorted(glob.glob(f'{atm_dir}/skytable_a_1.0_p_*.fits'))
	if not pwv_files:
		raise FileNotFoundError(
			f'No skytable files found in {atm_dir}. '
			f'Expected files named skytable_a_1.0_p_<PWV>.fits'
		)
	pwv_grid = np.array([
		float(f.split('_p_')[-1].replace('.fits', ''))
		for f in pwv_files
	])
	# Clip drawn PWV values to grid bounds
	p_list = np.clip(p_list, pwv_grid.min(), pwv_grid.max())

	# Cache loaded spectra to avoid re-reading the same file repeatedly
	_spec_cache = {}
	def _load_spec(pwv_val):
		if pwv_val not in _spec_cache:
			fname = f'{atm_dir}/skytable_a_1.0_p_{pwv_val}.fits'
			with fits.open(fname) as hdul:
				data = hdul[1].data
				_spec_cache[pwv_val] = (
					data['lam'] * 10,        # nm to Angstrom
					data['trans'].copy(),
					data['flux'].copy() * 1e-8,  # ph/s/cm^2/A
				)
		return _spec_cache[pwv_val]

	flux_startell = None
	wavlen_arr    = None
	snr_arr       = None

	# Progress bar — works without tqdm
	bar_width = 40
	def _print_progress(i, n):
		done  = int(bar_width * (i + 1) / n)
		bar   = '#' * done + '-' * (bar_width - done)
		print(f'\r  Building flux cube [{bar}] {i+1}/{n}', end='', flush=True)
		if i + 1 == n:
			print()

	for i in range(n_spectra):
		_print_progress(i, n_spectra)
		ppv = p_list[i]

		# Find bracketing grid points and interpolate in log(PWV) space
		# Log-space interpolation justified by Beer-Lambert law:
		# telluric absorption depth scales ~ linearly with log(PWV)
		idx_hi = np.searchsorted(pwv_grid, ppv)
		if idx_hi == 0:
			idx_hi = 1
		elif idx_hi >= len(pwv_grid):
			idx_hi = len(pwv_grid) - 1
		idx_lo = idx_hi - 1

		pwv_lo = pwv_grid[idx_lo]
		pwv_hi = pwv_grid[idx_hi]

		w_mod, tr_lo, sky_lo = _load_spec(pwv_lo)
		_,     tr_hi, sky_hi = _load_spec(pwv_hi)

		if pwv_lo == pwv_hi:
			w = 0.0
		else:
			w = ((np.log(ppv)    - np.log(pwv_lo)) /
			     (np.log(pwv_hi) - np.log(pwv_lo)))

		tr_mod = (1.0 - w) * tr_lo + w * tr_hi
		sky_fl = (1.0 - w) * sky_lo + w * sky_hi

		with np.errstate(divide='ignore', invalid='ignore'):
			tr_mod = np.exp(airm[i] * np.log(tr_mod))
		_, tr_con, fn_tr_con     = broaden(w_mod, tr_mod, res, splinefit=True)
		_, sky_con, fn_sky_con   = broaden(w_mod, sky_fl, res, splinefit=True)

		ph_det, noise, sn_met, wav_out, mask = signalcalc(
			inst, modes, t_exp, fn_star_ph, fn_tr_con, fn_sky_con
		)

		np.random.seed(seed_base * (i + 1))
		var     = np.random.uniform(low=0.95, high=1.05)
		ph_det  = ph_det * var

		if i == 0:
			if inst != 'andes':
				n_ord, n_pix    = ph_det.shape
				flux_startell   = np.zeros((n_spectra, n_ord, n_pix))
				wavlen_arr      = np.zeros((n_spectra, n_ord, n_pix))
				snr_arr         = np.zeros((n_spectra, n_ord, n_pix))
			else:
				n_pix           = len(ph_det)
				flux_startell   = np.zeros((n_spectra, n_pix))
				wavlen_arr      = np.zeros((n_spectra, n_pix))
				snr_arr         = np.zeros((n_spectra, n_pix))

		flux_startell[i] = ph_det
		wavlen_arr[i]    = wav_out
		snr_arr[i]       = sn_met

	if inst != 'andes':
		flux_cube_startell = np.swapaxes(flux_startell, 0, 1)
		wavlen_cube        = np.swapaxes(wavlen_arr,    0, 1)
		snr_cube           = np.swapaxes(snr_arr,       0, 1)
	else:
		flux_cube_startell = np.array([flux_startell])
		wavlen_cube        = np.array([wavlen_arr])
		snr_cube           = np.array([snr_arr])

	return flux_cube_startell, wavlen_cube, snr_cube, p_list


# ---------------------------------------------------------------------------
# Sub-function 6: Planet injection
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Planet model loader
# ---------------------------------------------------------------------------

def load_planet_model(path):
	"""
	Load a planet model spectrum from a file, detect its format, and return
	a summary so the user can identify the correct wavelength and flux arrays.

	Supported formats: .npy, .txt/.dat/.csv, .fits, .hdf5/.h5

	Parameters
	----------
	path : str   path to the planet model file

	Returns
	-------
	dict with keys depending on format:
	  For .npy:
	    'data'    : numpy.ndarray
	    'shape'   : tuple
	  For .txt/.dat/.csv:
	    'data'    : pandas.DataFrame
	    'columns' : list of str
	  For .fits:
	    'hdul'    : astropy.io.fits.HDUList (caller must close)
	    'info'    : str summary
	  For .hdf5/.h5:
	    'keys'    : list of str dataset keys
	    'file'    : h5py.File (caller must close)

	After inspecting the output, pass wav_angstrom and flux arrays to
	_inject_planet() directly.

	Example
	-------
	>>> m = load_planet_model('mymodel.npy')
	>>> wav  = m['data'][0] * 1e4   # row 0 = wavelength in micron -> Angstrom
	>>> flux = m['data'][1]         # row 1 = transit depth
	>>> results = ratri(..., planet_wav=wav, planet_flux=flux, ...)
	"""
	import os
	if not os.path.exists(path):
		raise FileNotFoundError(f'Planet model file not found: {path}')

	ext = os.path.splitext(path)[-1].lower()

	if ext == '.npy':
		data = np.load(path, allow_pickle=True)
		print(f'Loaded .npy file: shape={data.shape}, dtype={data.dtype}')
		print(f'  Row 0 range : {data[0].min():.4e} to {data[0].max():.4e}')
		if len(data) > 1:
			print(f'  Row 1 range : {data[1].min():.4e} to {data[1].max():.4e}')
		print('Identify which row/column is wavelength (in micron or Angstrom) and which is flux/depth.')
		return {'data': data, 'shape': data.shape}

	elif ext in ('.txt', '.dat', '.csv'):
		# Read with numpy, skipping comment lines starting with '#'
		# This avoids pandas misidentifying the comment header as column names
		# and handles trailing empty tab columns cleanly.
		try:
			data = np.loadtxt(path, comments='#')
		except Exception:
			# Fallback: try comma-separated
			try:
				data = np.loadtxt(path, comments='#', delimiter=',')
			except Exception as e:
				raise ValueError(f'Could not parse text file: {e}')

		# Parse column names from comment header if present
		col_names = []
		with open(path) as fh:
			for line in fh:
				if line.startswith('#'):
					# Strip '#' and split on tabs or multiple spaces
					parts = [p.strip() for p in line.lstrip('#').split('\t') if p.strip()]
					if len(parts) >= 2:
						col_names = parts
					break

		n_rows, n_cols = data.shape
		if not col_names or len(col_names) != n_cols:
			col_names = [f'col_{i}' for i in range(n_cols)]

		print(f'Loaded text file: {n_rows} rows x {n_cols} columns')
		for i, name in enumerate(col_names):
			print(f'  col {i}: {name:<40}  range [{data[:,i].min():.4e}, {data[:,i].max():.4e}]')

		# Auto-detect wavelength column (col 0) and flux column (col 1)
		# Convention: col 0 = wavelength, col 1 = total depth/flux ratio
		wav  = data[:, 0]
		flux = data[:, 1]

		# Detect wavelength unit from value range and convert to micron
		# so downstream code always receives micron.
		# Typical NIR ranges:
		#   cm      : 1e-5 to 3e-4  (petitRADTRANS default output)
		#   micron  : 0.3  to 30
		#   nm      : 300  to 3000
		#   Angstrom: 3000 to 30000
		wav_min, wav_max = wav.min(), wav.max()
		if wav_max < 0.1:
			# cm (petitRADTRANS: NIR range ~1e-5 to 3e-4 cm)
			wav_micron = wav * 1e4
			unit_str = 'cm -> converted to micron'
		elif wav_max > 10000:
			# Angstrom (typical NIR range 3000-30000 A)
			wav_micron = wav * 1e-4
			unit_str = 'Angstrom -> converted to micron'
		elif wav_max > 100:
			# nm (typical NIR range 300-3000 nm)
			wav_micron = wav * 1e-3
			unit_str = 'nm -> converted to micron'
		else:
			# micron (typical NIR range 0.3-30 um)
			wav_micron = wav
			unit_str = 'micron (no conversion needed)'

		print(f'\nAuto-selected: wavelength=col_0, flux/depth=col_1')
		print(f'  Wavelength unit detected: {unit_str}')
		print(f'  Wavelength range: {wav_micron.min():.4f} — {wav_micron.max():.4f} micron')
		print('If this is wrong, pass planet_wav and planet_flux manually to ratri().')

		return {
			'data'       : data,
			'columns'    : col_names,
			'planet_wav' : wav_micron,   # always in micron
			'planet_flux': flux,
		}

	elif ext == '.fits':
		from astropy.io import fits
		hdul = fits.open(path)
		print(f'Loaded FITS file:')
		hdul.info()
		for i, hdu in enumerate(hdul):
			if hasattr(hdu, 'columns') and hdu.columns:
				print(f'  Extension {i} columns: {hdu.columns.names}')
		print('Use hdul[ext].data to access the data. Remember to call hdul.close() when done.')
		return {'hdul': hdul, 'info': str(hdul.info())}

	elif ext in ('.hdf5', '.h5'):
		try:
			import h5py
		except ImportError:
			raise ImportError('h5py is required for HDF5 files. Run: pip install h5py')
		f = h5py.File(path, 'r')
		keys = list(f.keys())
		print(f'Loaded HDF5 file. Top-level keys: {keys}')
		for k in keys:
			ds = f[k]
			if hasattr(ds, 'shape'):
				print(f'  {k}: shape={ds.shape}, dtype={ds.dtype}')
		print('Use file[key][()] to read a dataset as a numpy array. Remember to close the file.')
		return {'keys': keys, 'file': f}

	else:
		raise ValueError(
			f'Unrecognised file extension "{ext}". '
			f'Supported: .npy, .txt, .dat, .csv, .fits, .hdf5, .h5'
		)


def _inject_planet(flux_cube_startell, wavlen_cube, RVs_t, transit_mask,
                   res, ob, tr_duration_h, period_days, ph,
                   planet_wav=None, planet_flux=None,
                   noise_seed=3):
	"""
	Broaden, Doppler shift, and inject a planet model into the flux cube.
	If planet_wav and planet_flux are both None, skips injection and returns
	the star-only cube with Poisson noise.

	Call load_planet_model() first to inspect your model file and extract
	the correct wavelength and flux arrays, then pass them here.

	Parameters
	----------
	flux_cube_startell : numpy.ndarray  (n_orders, n_spectra, n_pixels)
	wavlen_cube        : numpy.ndarray  (n_orders, n_spectra, n_pixels)
	RVs_t              : numpy.ndarray  planet RVs in km/s
	transit_mask       : numpy.ndarray of bool
	res                : float spectral resolution
	ob                 : str   'tm' or 'em'
	tr_duration_h      : float transit duration in hours
	period_days        : float orbital period in days
	ph                 : numpy.ndarray orbital phases
	planet_wav         : numpy.ndarray or None
	                     wavelength array in Angstrom
	planet_flux        : numpy.ndarray or None
	                     flux/depth array (same length as planet_wav)
	noise_seed         : int   random seed for white noise draw

	Returns
	-------
	flux_cube     : numpy.ndarray  noisy star-only or star+planet cube
	flux_err_cube : numpy.ndarray  noise realisation
	"""
	if planet_wav is None or planet_flux is None:
		print('  No planet model provided — returning star-only cube with noise.')
		noise_rms     = flux_cube_startell ** 0.5
		np.random.seed(noise_seed)
		flux_err_cube = np.random.normal(np.zeros_like(flux_cube_startell), noise_rms)
		flux_cube     = flux_cube_startell + flux_err_cube
		return flux_cube, flux_err_cube

	wav_p    = np.asarray(planet_wav)
	p_flux   = np.asarray(planet_flux)

	# No wavelength clip — use the full range provided by the model.
	# broaden/splinefit will only be evaluated at instrument wavelengths anyway.
	wavlen_p = wav_p
	p_srflux = p_flux

	wreg, psr_con = broaden(wavlen_p, p_srflux, res, splinefit=False)

	if ob == 'em':
		psr_flip_con_c = 1.0 + psr_con
		in_event       = transit_mask

	if ob == 'tm':
		import pandas as pd
		psr_series      = pd.Series(1.0 + np.negative(psr_con))
		psr_rollmax     = psr_series.rolling(500, 1).max()
		psr_flip_con_c  = (psr_series / psr_rollmax).to_numpy()

		half_dur        = 0.5 * tr_duration_h / (period_days * 24.0)
		in_event        = np.abs(ph) <= half_dur

	f_p              = splinefit(wreg, psr_flip_con_c)
	n_orders, n_spectra, n_pixels = flux_cube_startell.shape
	RVs_event        = RVs_t[in_event]

	flux_cube_st_pl  = modelinjectdatacube(
		RVs_event, n_orders, n_spectra, n_pixels,
		flux_cube_startell, wavlen_cube, f_p, in_event
	)

	noise_rms = flux_cube_st_pl ** 0.5
	np.random.seed(noise_seed)
	flux_err_cube = np.random.normal(np.zeros_like(flux_cube_st_pl), noise_rms)
	flux_cube     = flux_cube_st_pl + flux_err_cube

	return flux_cube, flux_err_cube


# ---------------------------------------------------------------------------
# Sub-function 7: Save outputs and diagnostic plots
# ---------------------------------------------------------------------------

def _save_and_plot(savedir, inst, ob, flux_cube, flux_cube_startell,
                   flux_err_cube, wavlen_cube, snr_cube,
                   bjd, barycorr_kms, ph, airm, pwv_array):
	"""
	Save all output cubes and produce diagnostic phase/airmass/altitude plots.

	Parameters
	----------
	savedir            : str   output directory (created if absent)
	inst               : str
	flux_cube          : numpy.ndarray
	flux_cube_startell : numpy.ndarray
	flux_err_cube      : numpy.ndarray
	wavlen_cube        : numpy.ndarray
	snr_cube           : numpy.ndarray
	bjd                : numpy.ndarray
	barycorr_kms       : numpy.ndarray
	ph                 : numpy.ndarray   orbital phases
	airm               : numpy.ndarray   airmass per exposure
	"""
	import os
	os.makedirs(savedir, exist_ok=True)

	np.save(f'{savedir}flux_cube_{inst}.npy',          flux_cube)
	np.save(f'{savedir}flux_cube_startell_{inst}.npy', flux_cube_startell)
	np.save(f'{savedir}flux_err_cube_{inst}.npy',      flux_err_cube)
	np.save(f'{savedir}wavlen_cube_{inst}.npy',        wavlen_cube)
	np.save(f'{savedir}snr_cube_{inst}.npy',           snr_cube)
	np.save(f'{savedir}bjd.npy',                       bjd)
	np.save(f'{savedir}bar_rad_vel.npy',               barycorr_kms)
	np.save(f'{savedir}pwv.npy',                       pwv_array)
	np.save(f'{savedir}airmass.npy',                   airm)
	np.save(f'{savedir}phases.npy',                    ph)

	print(f'Saved cubes to {savedir}')

	# Phase axis for plotting:
	#   emission: remap to [0,1) via %1 — occultation at 0.5, no discontinuity
	#   transmission: use raw phases [-0.5, 0.5) — transit at 0, no discontinuity
	if ob == 'em':
		ph_plot    = ph % 1.0
		ph_xlabel  = r'Orbital Phase $\phi$ [0, 1)'
		ph_xlim    = (0, 1)
	else:
		ph_plot    = ph
		ph_xlabel  = r'Orbital Phase $\phi$ [-0.5, 0.5)'
		ph_xlim    = (-0.5, 0.5)

	# Phase vs airmass
	plt.figure()
	plt.plot(ph_plot, airm)
	plt.scatter(ph_plot, airm, s=8)
	plt.xlabel(ph_xlabel)
	plt.ylabel('Airmass')
	plt.xlim(*ph_xlim)
	plt.tight_layout()
	plt.savefig(f'{savedir}airmphase_{inst}.png', dpi=200, bbox_inches='tight')
	plt.close()

	# All orders flux image in a single figure
	n_orders = flux_cube.shape[0]
	fig, axes = plt.subplots(n_orders, 1, figsize=(12, 2 * n_orders), sharex=True)
	if n_orders == 1:
		axes = [axes]
	for oi, ax in enumerate(axes):
		im = ax.imshow(
			flux_cube[oi],
			aspect='auto',
			origin='lower',
			interpolation='none',
		)
		ax.set_ylabel(f'Ord {oi}', fontsize=7)
		plt.colorbar(im, ax=ax, pad=0.01)
	axes[-1].set_xlabel('Pixel')
	fig.suptitle(f'All orders — {inst}', fontsize=10)
	plt.tight_layout(rect=[0, 0, 1, 0.97])
	plt.savefig(f'{savedir}flux_allorders_{inst}.png', dpi=150, bbox_inches='tight')
	plt.close()


# ---------------------------------------------------------------------------
# Top-level: ratri()
# ---------------------------------------------------------------------------

def ratri(
	target_name,
	inst,
	modes,
	ob,
	t_exp,
	T0,
	period,
	mean_transit_BJD,
	e,
	omega,
	vsys,
	Kp,
	d_pc,
	R_st_rsun,
	tr_duration_h,
	t_search_start,
	t_search_end,
	phase_min,
	phase_max,
	phoenix_spec_path,
	phoenix_wave_path,
	savedir,
	planet_model_path          = None,
	qual               = None,
	atm_dir            = './atm_transmission',
	airmass_min        = 1.0,
	airmass_max        = 2.0,
	sun_horizon        = -18.0,
	min_duration       = 1.0,
	night_rank         = 0,
	noise_seed         = 3,
	chosen_night       = None,
):
	"""
	Automated night selection and observation simulator.

	Searches a date range for the best observable nights for the requested
	orbital phase window, selects one (by rank), builds a simulated
	observation including stellar spectrum, telluric absorption, and
	injected planet signal, then saves all output cubes.

	Parameters
	----------
	target_name       : str   Simbad-resolvable target name, e.g. 'V1298 Tau'
	inst              : str   instrument key: 'crires+', 'carmenes', 'giano',
	                          'spirou', 'andes', 'andes_ccd_carmenes'
	modes             : list  instrument mode list (see signalcalc)
	ob                : str   'tm' (transit) or 'em' (emission/occultation)
	t_exp             : float exposure time in seconds
	qual              : str   telluric quality: 'good', 'avg', or 'bad'
	T0                : astropy.time.Time   mid-transit epoch (BJD_TDB)
	period            : astropy.units.Quantity   orbital period
	mean_transit_BJD  : float BJD_TDB of reference transit midpoint
	e                 : float eccentricity
	omega             : float argument of periastron in radians
	vsys              : float systemic velocity in km/s
	Kp                : float planet RV semi-amplitude in km/s
	d_pc              : float distance to star in parsec
	R_st_rsun         : float stellar radius in solar radii
	tr_duration_h     : float transit duration in hours
	t_search_start    : astropy.time.Time   start of date range to search
	t_search_end      : astropy.time.Time   end of date range to search
	phase_min         : float lower phase bound in [-0.5, 0.5)
	phase_max         : float upper phase bound; phase_min > phase_max wraps
	phoenix_spec_path : str   path to PHOENIX flux FITS file
	phoenix_wave_path : str   path to PHOENIX wavelength FITS file
	planet_model_path : str   path to planet model text file
	savedir           : str   output directory
	atm_dir           : str   directory with skytable_a_1.0_p_*.fits files
	airmass_min       : float (default 1.0)
	airmass_max       : float (default 2.0)
	sun_horizon       : float sun altitude threshold in degrees (default -18, astronomical twilight)
	min_duration      : float minimum hours in phase window per night (default 1.0)
	night_rank        : int   index into sorted night list to simulate (default 0 = best)
	noise_seed        : int   random seed for white noise draw (default 3)
	"""
	params = _INSTRUMENT_PARAMS[inst]
	res    = params['res']

	# Resolve qual: use site default if not provided
	if qual is None:
		qual = _OBSERVATORIES[inst]['default_qual']
		print(f'  PWV quality not specified — using site default: {qual}')

	# 1. Find qualifying nights — skip if caller already provides the chosen night
	if chosen_night is not None:
		chosen = chosen_night
		print(f'\nUsing pre-selected night: {chosen["night"]}')
	else:
		nights = _select_night(
			target_name    = target_name,
			inst           = inst,
			t_start        = t_search_start,
			t_end          = t_search_end,
			T0             = T0,
			period         = period,
			phase_min      = phase_min,
			phase_max      = phase_max,
			airmass_min    = airmass_min,
			airmass_max    = airmass_max,
			sun_horizon    = sun_horizon,
			min_duration   = min_duration,
		)

		if not nights:
			print('No qualifying nights found. Adjust search parameters.')
			return None

		chosen = nights[night_rank]
		print(f'\nSimulating night {night_rank + 1} of {len(nights)}: {chosen["night"]}')

	# 2. Build time grid from the observable window found by _select_night
	observe_time, n_spectra = _build_time_grid(
		inst      = inst,
		obs_start = chosen['obs_start'],
		obs_end   = chosen['obs_end'],
		t_exp     = t_exp,
	)

	# 3. Orbital quantities
	orbit = _compute_orbit(
		observe_time     = observe_time,
		inst             = inst,
		target_name      = target_name,
		T0               = T0,
		period           = period,
		mean_transit_BJD = mean_transit_BJD,
		e                = e,
		omega            = omega,
		vsys             = vsys,
		Kp               = Kp,
		ob               = ob,
		tr_duration_h    = tr_duration_h,
	)

	# 4. Stellar spectrum
	fn_star_ph, _ = _load_stellar_spectrum(
		phoenix_spec_path = phoenix_spec_path,
		phoenix_wave_path = phoenix_wave_path,
		res               = res,
		d_pc              = d_pc,
		R_st_rsun         = R_st_rsun,
	)

	# 5. Flux cube
	# Use MJD of first exposure as seed so each night gets different
	# PWV draws but results are reproducible for the same night.
	night_seed = int(orbit['bjd'][0]) % 100000
	flux_cube_startell, wavlen_cube, snr_cube, pwv_array = _build_flux_cube(
		inst       = inst,
		modes      = modes,
		t_exp      = t_exp,
		res        = res,
		n_spectra  = n_spectra,
		airm       = orbit['airmass'],
		fn_star_ph = fn_star_ph,
		qual       = qual,
		atm_dir    = atm_dir,
		seed_base  = night_seed,
	)

	# 6. Planet injection
	# Load model if path provided, otherwise pass None arrays (star-only)
	planet_wav  = None
	planet_flux = None
	if planet_model_path is not None:
		print('\nLoading planet model...')
		print('If the auto-detection is wrong, call load_planet_model() manually')
		print('and pass planet_wav and planet_flux to ratri() directly instead.')
		model       = load_planet_model(planet_model_path)
		# Use pre-extracted arrays if available (txt/dat/csv with auto-detection)
		if 'planet_wav' in model and 'planet_flux' in model:
			planet_wav  = model['planet_wav'] * 1e4   # micron to Angstrom
			planet_flux = model['planet_flux']
		elif 'shape' in model:
			# .npy: assume row 0 = wavelength, row 1 = flux
			# Detect unit from range and convert to Angstrom
			_wav = model['data'][0]
			if _wav.max() < 0.1:
				planet_wav = _wav * 1e8        # cm to Angstrom
			elif _wav.max() > 10000:
				planet_wav = _wav              # already Angstrom
			elif _wav.max() > 100:
				planet_wav = _wav * 10         # nm to Angstrom
			else:
				planet_wav = _wav * 1e4        # micron to Angstrom
			planet_flux = model['data'][1]
		else:
			print('  Cannot auto-load this format. Call load_planet_model() manually.')

	flux_cube, flux_err_cube = _inject_planet(
		flux_cube_startell = flux_cube_startell,
		wavlen_cube        = wavlen_cube,
		RVs_t              = orbit['RVs_t'],
		transit_mask       = orbit['transit_mask'],
		res                = res,
		ob                 = ob,
		tr_duration_h      = tr_duration_h,
		period_days        = period.to(u.day).value,
		ph                 = orbit['ph'],
		planet_wav         = planet_wav,
		planet_flux        = planet_flux,
		noise_seed         = noise_seed,
	)

	# 7. Save and plot
	_save_and_plot(
		savedir            = savedir,
		inst               = inst,
		ob                 = ob,
		flux_cube          = flux_cube,
		flux_cube_startell = flux_cube_startell,
		flux_err_cube      = flux_err_cube,
		wavlen_cube        = wavlen_cube,
		snr_cube           = snr_cube,
		bjd                = orbit['bjd'],
		barycorr_kms       = orbit['barycorr_kms'],
		ph                 = orbit['ph'],
		airm               = orbit['airmass'],
		pwv_array          = pwv_array,
	)

	return dict(
		nights             = nights if chosen_night is None else None,
		chosen_night       = chosen,
		observe_time       = observe_time,
		orbit              = orbit,
		flux_cube          = flux_cube,
		flux_cube_startell = flux_cube_startell,
		flux_err_cube      = flux_err_cube,
		wavlen_cube        = wavlen_cube,
		snr_cube           = snr_cube,
	)


# ===========================================================================
# ratri_auto() — ratri() with automated stellar parameter lookup and
#               spectrum download via stellar.py
#
# Drop-in replacement for ratri() that takes a planet name instead of
# manual phoenix_spec_path / phoenix_wave_path arguments.
# ===========================================================================

def ratri_auto(
	planet_name,
	inst,
	modes,
	ob,
	t_exp,
	T0,
	period,
	mean_transit_BJD,
	e,
	omega,
	vsys,
	Kp,
	tr_duration_h,
	t_search_start,
	t_search_end,
	phase_min,
	phase_max,
	savedir,
	spectra_cache_dir,
	planet_model_path          = None,
	star_name          = None,
	library            = None,
	atm_dir            = './atm_transmission',
	airmass_min        = 1.0,
	airmass_max        = 2.0,
	sun_horizon        = -18.0,
	min_duration       = 1.0,
	night_rank         = 0,
	noise_seed         = 3,
	chosen_night       = None,
	qual               = None,
):
	"""
	Automated version of ratri() that resolves stellar parameters from
	the planet name and downloads the appropriate atmosphere model.

	Parameters
	----------
	planet_name       : str   planet name, e.g. '55 Cnc e', 'HD 189733 b'
	                          Used for NASA Exoplanet Archive lookup and
	                          also as the target for coordinate resolution.
	star_name         : str or None
	                          Host star name for SIMBAD fallback, e.g. '55 Cnc'.
	                          If None, planet_name is used.
	spectra_cache_dir : str   directory to cache downloaded spectrum files
	library           : str or None
	                          Force a specific atmosphere library:
	                          'phoenix', 'btsettl', 'atlas9', 'marcs'.
	                          If None, auto-recommended based on T_eff.

	All other parameters are identical to ratri(). The d_pc and R_st_rsun
	arguments are retrieved automatically from the NASA Exoplanet Archive
	or left as None if unavailable (in which case a warning is printed and
	defaults are used).

	Returns
	-------
	Same as ratri(): dict with nights, orbit, flux cubes, etc.
	Plus additional keys:
	  stellar_params : dict   raw looked-up parameters
	  spectrum_paths : dict   paths to downloaded spectrum files
	"""
	from stellar import get_spectrum_for_planet

	# 1. Look up stellar parameters and download spectrum
	target_for_lookup = star_name if star_name else planet_name
	stellar_params, spectrum_paths = get_spectrum_for_planet(
		planet_name = planet_name,
		cache_dir   = spectra_cache_dir,
		star_name   = star_name,
		library     = library,
	)

	d_pc      = stellar_params.get('distance')
	R_st_rsun = stellar_params.get('radius')

	if d_pc is None:
		print('  Distance not found in archive. Defaulting to 100 pc — override via d_pc if known.')
		d_pc = 100.0
	if R_st_rsun is None:
		print('  Stellar radius not found in archive. Defaulting to 1.0 R_sun — override via R_st_rsun if known.')
		R_st_rsun = 1.0

	# Resolve the target name for phase/orbit calculations.
	# For planet names like '55 Cnc e', SkyCoord.from_name resolves the host
	# star since the planet itself has no resolvable coordinates.
	target_name = star_name if star_name else planet_name.rsplit(' ', 1)[0]

	# 2. Delegate to ratri() with resolved paths
	results = ratri(
		target_name       = target_name,
		inst              = inst,
		modes             = modes,
		ob                = ob,
		t_exp             = t_exp,
		qual              = qual,
		T0                = T0,
		period            = period,
		mean_transit_BJD  = mean_transit_BJD,
		e                 = e,
		omega             = omega,
		vsys              = vsys,
		Kp                = Kp,
		d_pc              = d_pc,
		R_st_rsun         = R_st_rsun,
		tr_duration_h     = tr_duration_h,
		t_search_start    = t_search_start,
		t_search_end      = t_search_end,
		phase_min         = phase_min,
		phase_max         = phase_max,
		phoenix_spec_path = spectrum_paths['spec_path'],
		phoenix_wave_path = spectrum_paths['wave_path'],
		planet_model_path = planet_model_path,
		savedir           = savedir,
		atm_dir           = atm_dir,
		airmass_min       = airmass_min,
		airmass_max       = airmass_max,
		sun_horizon       = sun_horizon,
		min_duration      = min_duration,
		night_rank        = night_rank,
		noise_seed        = noise_seed,
		chosen_night      = chosen_night,
	)

	if results is None:
		print('No results returned — no qualifying nights found. Adjust search parameters.')
		return None

	results['stellar_params'] = stellar_params
	results['spectrum_paths'] = spectrum_paths

	return results



# ---------------------------------------------------------------------------
# Preview / planning tool
# ---------------------------------------------------------------------------

def ratri_preview_setup(
	planet_name,
	inst,
	modes,
	ob,
	t_exp,
	qual              = None,
	airmass           = 1.2,
	sn_mask_threshold = 80,
	atm_dir           = './atm_transmission',
	spectra_cache_dir = './spectra_cache',
	star_name         = None,
	library           = None,
	savedir           = './ratri_preview_output',
):
	"""
	Setup step for ratri_preview — loads stellar spectrum, skytable, and
	computes star-only SNR. Call once per instrument/star/atmosphere
	combination, then pass the returned context dict to ratri_preview_ccf
	for each molecule.

	Parameters
	----------
	planet_name       : str
	inst              : str   instrument key from _OBSERVATORIES
	modes             : list
	ob                : str   'tm' or 'em'
	t_exp             : float exposure time in seconds
	qual              : str or None   PWV quality tier; None = site default
	airmass           : float airmass for telluric correction (default 1.2)
	sn_mask_threshold : float SNR mask threshold (default 80)
	atm_dir           : str
	spectra_cache_dir : str
	star_name         : str or None
	library           : str or None
	savedir           : str   output directory for figures

	Returns
	-------
	ctx : dict with keys:
	  'inst', 'modes', 'ob', 't_exp', 'res', 'qual', 'pwv_med', 'airmass',
	  'fn_star_ph', 'fn_tr_con', 'fn_sky_con',
	  'ph_det_st', 'noise_st', 'sn_met', 'wav', 'mask_st',
	  'ph_det_st_safe', 'sn_mask_threshold', 'savedir', 'planet_name'
	"""
	import os
	import glob
	from astropy.io import fits

	os.makedirs(savedir, exist_ok=True)

	# Resolve qual
	if qual is None:
		qual = _OBSERVATORIES[inst]['default_qual']
		print(f'PWV quality not specified — using site default: {qual}')

	pwv_med = _PWV_TIERS[qual][1]
	print(f'Using PWV = {pwv_med} mm (median of {qual} tier), airmass = {airmass}')

	# Instrument resolution
	res = _INSTRUMENT_PARAMS[inst]['res']

	# Load stellar spectrum
	print('\nLoading stellar spectrum...')
	from stellar import get_spectrum_for_planet
	stellar_params, spectrum_paths = get_spectrum_for_planet(
		planet_name, spectra_cache_dir,
		star_name=star_name, library=library,
	)
	d_pc      = stellar_params['distance']
	R_st_rsun = stellar_params.get('radius', 1.0)

	fn_star_ph, wav_stellar = _load_stellar_spectrum(
		spectrum_paths['spec_path'],
		spectrum_paths['wave_path'],
		res, d_pc, R_st_rsun,
	)

	# Load and interpolate skytable at median PWV
	print(f'\nLoading skytable at PWV = {pwv_med} mm...')
	pwv_files = sorted(glob.glob(f'{atm_dir}/skytable_a_1.0_p_*.fits'))
	if not pwv_files:
		raise FileNotFoundError(f'No skytable files found in {atm_dir}')
	pwv_grid = np.array([
		float(f.split('_p_')[-1].replace('.fits', ''))
		for f in pwv_files
	])
	pwv_clipped = np.clip(pwv_med, pwv_grid.min(), pwv_grid.max())
	idx_hi = int(np.searchsorted(pwv_grid, pwv_clipped))
	if idx_hi == 0:
		idx_hi = 1
	elif idx_hi >= len(pwv_grid):
		idx_hi = len(pwv_grid) - 1
	idx_lo = idx_hi - 1

	def _load_sky(pwv_val):
		with fits.open(f'{atm_dir}/skytable_a_1.0_p_{pwv_val}.fits') as hdul:
			d = hdul[1].data
			return d['lam'] * 10, d['trans'].copy(), d['flux'].copy() * 1e-8

	w_mod, tr_lo, sky_lo = _load_sky(pwv_grid[idx_lo])
	_,     tr_hi, sky_hi = _load_sky(pwv_grid[idx_hi])

	if pwv_grid[idx_lo] == pwv_grid[idx_hi]:
		w = 0.0
	else:
		w = ((np.log(pwv_clipped)       - np.log(pwv_grid[idx_lo])) /
		     (np.log(pwv_grid[idx_hi])  - np.log(pwv_grid[idx_lo])))

	tr_mod  = (1 - w) * tr_lo + w * tr_hi
	sky_mod = (1 - w) * sky_lo + w * sky_hi

	with np.errstate(divide='ignore', invalid='ignore'):
		tr_mod = np.exp(airmass * np.log(tr_mod))

	_, tr_con, fn_tr_con   = broaden(w_mod, tr_mod,  res, splinefit=True)
	_, sky_con, fn_sky_con = broaden(w_mod, sky_mod, res, splinefit=True)

	# Star-only SNR
	print('\nComputing star-only SNR...')
	ph_det_st, noise_st, sn_met, wav, mask_st = signalcalc(
		inst, modes, t_exp, fn_star_ph, fn_tr_con, fn_sky_con
	)
	ph_det_st_safe = np.where(ph_det_st < 1, 1.0, ph_det_st)

	return {
		'planet_name'       : planet_name,
		'inst'              : inst,
		'modes'             : modes,
		'ob'                : ob,
		't_exp'             : t_exp,
		'res'               : res,
		'qual'              : qual,
		'pwv_med'           : pwv_med,
		'airmass'           : airmass,
		'fn_star_ph'        : fn_star_ph,
		'fn_tr_con'         : fn_tr_con,
		'fn_sky_con'        : fn_sky_con,
		'ph_det_st'         : ph_det_st,
		'ph_det_st_safe'    : ph_det_st_safe,
		'noise_st'          : noise_st,
		'sn_met'            : sn_met,
		'wav'               : wav,
		'mask_st'           : mask_st,
		'sn_mask_threshold' : sn_mask_threshold,
		'savedir'           : savedir,
		'wav_stellar'       : wav_stellar,  # broadened stellar wavelength grid
	}


def ratri_preview_snr(ctx):
	"""
	Plot and save the SNR figure from a setup context.
	Call once after ratri_preview_setup.

	Parameters
	----------
	ctx : dict returned by ratri_preview_setup

	Returns
	-------
	path : str   path to saved figure
	"""
	import os
	import re
	inst       = ctx['inst']
	wav        = ctx['wav']
	sn_met     = ctx['sn_met']
	pwv_med    = ctx['pwv_med']
	airmass    = ctx['airmass']
	planet_name= ctx['planet_name']
	ob         = ctx['ob']
	t_exp      = ctx['t_exp']
	savedir    = ctx['savedir']

	safe_name  = re.sub(r'[^a-zA-Z0-9_]', '_', planet_name).strip('_')
	snr_path   = os.path.join(
		savedir, f'{safe_name}_{inst}_{ob}_{t_exp}s_snr.png'
	)

	fig, ax = plt.subplots(figsize=(12, 4))
	if inst == 'andes' or wav.ndim == 1:
		ax.plot(wav, sn_met, color='steelblue', alpha=0.7, lw=0.8)
	else:
		for oi in range(wav.shape[0]):
			ax.plot(wav[oi], sn_met[oi], color='steelblue', alpha=0.5, lw=0.6)
	for thresh in [50, 100, 200]:
		ax.axhline(thresh, color='gray', ls='--', lw=0.7, alpha=0.6)
		ax.text(wav.max() * 1.001, thresh, str(thresh),
		        va='center', fontsize=7, color='gray')
	ax.set_xlabel(r'Wavelength ($\AA$)')
	ax.set_ylabel('SNR per pixel')
	ax.set_title(
		f'{planet_name} — {inst.upper()} — {ob.upper()} — '
		f't_exp={t_exp}s — PWV={pwv_med}mm — airmass={airmass}'
	)
	ax.set_xlim(wav.min(), wav.max())
	ax.set_ylim(bottom=0)
	fig.tight_layout()
	fig.savefig(snr_path, dpi=200, bbox_inches='tight')
	print(f'SNR plot saved to {snr_path}')
	plt.show()
	plt.close(fig)
	return snr_path


def ratri_preview_ccf(
	ctx,
	wav_p,
	tr_d,
	tr_d_only,
	mol_label         = 'molecule',
	rv_centre         = 0.0,
	rv_left           = 100.0,
	rv_right          = 100.0,
	rv_inject         = 0.0,
	damp_values       = None,
	n_damp_auto       = 6,
	noise_seed        = 5,
	rv_buffer         = 20.0,
):
	"""
	CCF and detectability step for ratri_preview — the per-molecule loop.

	Parameters
	----------
	ctx         : dict returned by ratri_preview_setup
	wav_p       : numpy.ndarray   planet wavelengths in Angstrom
	tr_d        : numpy.ndarray   total transit depth or Fp/Fs (col 1)
	tr_d_only   : numpy.ndarray   molecule-only depth or flux (col 3)
	mol_label   : str   molecule name for plot titles and filenames
	rv_centre   : float km/s
	rv_left     : float km/s left of centre
	rv_right    : float km/s right of centre
	damp_values : list or None
	n_damp_auto : int
	noise_seed  : int
	rv_buffer   : float km/s — RV range around rv_centre excluded when
	              computing the noise floor for detection significance

	Returns
	-------
	dict with keys:
	  'mol_label', 'rv_grid', 'ccf', 'logl', 'sigma',
	  'damp_values', 'det_sigma'
	  where det_sigma[di] = median(sigma_wings) - min(sigma) for each damp
	"""
	import os
	import re
	from scipy import stats

	inst              = ctx['inst']
	modes             = ctx['modes']
	ob                = ctx['ob']
	t_exp             = ctx['t_exp']
	res               = ctx['res']
	fn_star_ph        = ctx['fn_star_ph']
	fn_tr_con         = ctx['fn_tr_con']
	fn_sky_con        = ctx['fn_sky_con']
	ph_det_st_safe    = ctx['ph_det_st_safe']
	sn_mask_threshold = ctx['sn_mask_threshold']
	wav               = ctx['wav']
	savedir           = ctx['savedir']
	planet_name       = ctx['planet_name']
	pwv_med           = ctx['pwv_med']

	print(f'\n--- {mol_label} (injected at RV={rv_inject} km/s) ---')

	# Injection uses shifted wav_p (planet at rv_inject in stellar frame)
	# Template uses UNSHIFTED wav_p so the CCF loop finds the match
	# at rv=rv_inject when sweeping over RV shifts
	wav_p_shifted = doppler(wav_p, rv_inject)
	wav_st = ctx['wav_stellar']

	if ob == 'em':
		# Injection on shifted grid
		wreg_inj, fp_fs_inj = broaden(wav_p_shifted, tr_d, res)
		fn_p_inj        = splinefit(wreg_inj, fp_fs_inj)
		pl_flux_st      = splev(wav_st, fn_p_inj, der=0)
		star_flux_st    = splev(wav_st, fn_star_ph, der=0)
		tot_flux_ph_arr = (1.0 + pl_flux_st) * star_flux_st
		fn_pl_star_ph   = splinefit(wav_st, tot_flux_ph_arr)
		# Template on unshifted grid for CCF
		wreg_pmod, fp_fs_only_broad = broaden(wav_p, tr_d_only, res)
		_,         fp_fs_broad      = broaden(wav_p, tr_d,       res)
	else:
		# Injection on shifted grid
		wreg_inj, tr_d_inj = broaden(wav_p_shifted, tr_d, res)
		fn_p_inj        = splinefit(wreg_inj, tr_d_inj)
		pl_flux_st      = splev(wav_st, fn_p_inj, der=0)
		star_flux_st    = splev(wav_st, fn_star_ph, der=0)
		tot_flux_ph_arr = (1.0 - pl_flux_st) * star_flux_st
		fn_pl_star_ph   = splinefit(wav_st, tot_flux_ph_arr)
		# Template on unshifted grid for CCF
		wreg_pmod, tr_d_only_broad = broaden(wav_p, tr_d_only, res)
		_,         tr_d_broad      = broaden(wav_p, tr_d,       res)

	# Star+planet SNR
	ph_det_comb, noise_comb, sn_met_comb, wav, mask_comb = signalcalc(
		inst, modes, t_exp, fn_pl_star_ph, fn_tr_con, fn_sky_con
	)
	mask_sn = sn_met_comb < sn_mask_threshold

	# Build CCF template -- no continuum normalisation
	# The injected signal preserves absolute depths, so the template
	# should match the data without additional normalisation.
	# Rolling normalisation would remove broadband information that
	# is still present in the data residuals without Gaussian filtering.
	if ob == 'em':
		template = fp_fs_only_broad
	else:
		template = 1.0 - tr_d_only_broad

	# RV grid
	rv_grid = np.arange(rv_centre - rv_left,
	                    rv_centre + rv_right + 1.0, 1.0)
	n_rv    = len(rv_grid)

	# n_orders
	n_orders = 1 if (inst == 'andes' or wav.ndim == 1) else wav.shape[0]

	# Damp values
	if damp_values is None:
		damp_values = sorted(set(
			int(round(v)) for v in np.geomspace(1, 30, n_damp_auto)
		))
	n_damp = len(damp_values)
	print(f'Damp values: {damp_values}')

	ccf_all   = np.zeros((n_damp, n_rv))
	logl_all  = np.zeros((n_damp, n_rv))
	sigma_all = np.zeros((n_damp, n_rv))
	det_sigma = np.zeros(n_damp)

	# Wing mask for detection significance (exclude rv_buffer around centre)
	wing_mask = np.abs(rv_grid - rv_centre) > rv_buffer

	for di, damp in enumerate(damp_values):
		np.random.seed(noise_seed + di)
		wh_noise_d  = (1.0 / damp) * np.random.normal(
			np.zeros_like(ph_det_comb), noise_comb
		)
		ph_noisy_d  = ph_det_comb + wh_noise_d
		mask_add_d  = ph_noisy_d < 1
		ph_noisy_d[ph_noisy_d < 1] = 1
		mask_all_d  = mask_comb | mask_add_d | mask_sn
		norm_flux_d = np.ma.array(
			ph_noisy_d / ph_det_st_safe, mask=mask_all_d
		).filled(fill_value=1.0)

		ccf_mat  = np.zeros((n_orders, n_rv))
		logl_mat = np.zeros((n_orders, n_rv))

		for ji, rv in enumerate(rv_grid):
			w_in     = doppler(wreg_pmod, rv)
			fn_pl    = splinefit(w_in, template)
			flux_new = splev(wav, fn_pl, der=0)

			if n_orders == 1:
				ccf_mat[0, ji], logl_mat[0, ji] = cross_corr_logl(
					norm_flux_d if wav.ndim == 1 else norm_flux_d[0],
					flux_new if wav.ndim == 1 else flux_new[0],
					mask_all_d if wav.ndim == 1 else mask_all_d[0],
				)
			else:
				for oi in range(n_orders):
					ccf_mat[oi, ji], logl_mat[oi, ji] = cross_corr_logl(
						norm_flux_d[oi], flux_new[oi], mask_all_d[oi]
					)

		ccf_sum  = np.sum(ccf_mat,  axis=0)
		logl_sum = np.sum(logl_mat, axis=0)

		lmax  = np.max(logl_sum)
		chi2  = 2.0 * (lmax - logl_sum)
		p_val = stats.chi2.sf(chi2, 1)
		p_val = np.where(p_val <= 0, 1e-300, p_val)
		sig   = stats.norm.isf(p_val / 2.0)

		ccf_all[di]   = ccf_sum
		logl_all[di]  = logl_sum
		sigma_all[di] = sig

		# Detection significance = noise floor - trough
		noise_floor   = np.median(sig[wing_mask]) if wing_mask.any() else sig.max()
		det_sigma[di] = noise_floor - sig.min()
		print(f'  damp={damp:>3}  (N={damp**2:>4})  '
		      f'floor={noise_floor:.1f}  trough={sig.min():.1f}  '
		      f'det_sigma={det_sigma[di]:.1f}')

	# --- Plot detectability figure ---
	safe_name  = re.sub(r'[^a-zA-Z0-9_]', '_', planet_name).strip('_')
	safe_mol   = re.sub(r'[^a-zA-Z0-9_]', '_', mol_label).strip('_')
	fig_path   = os.path.join(
		savedir,
		f'{safe_name}_{ctx["inst"]}_{ob}_{t_exp}s_{safe_mol}_detectability.png'
	)

	colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_damp))
	fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

	for di, damp in enumerate(damp_values):
		label = f'damp={damp} (N={damp**2})'
		axes[0].plot(rv_grid, ccf_all[di],   color=colors[di], label=label, lw=1.2)
		axes[1].plot(rv_grid, sigma_all[di], color=colors[di], lw=1.2)

	for ax in axes:
		ax.axvline(rv_centre, color='k', ls=':', lw=0.8, alpha=0.5)
		ax.axvline(rv_inject, color='gray', ls='--', lw=0.8, alpha=0.6,
		           label=f'injected RV={rv_inject} km/s' if ax is axes[0] else None)
	axes[1].axhline(3, color='red',     ls='--', lw=0.8, alpha=0.7, label=r'3$\sigma$')
	axes[1].axhline(5, color='darkred', ls='--', lw=0.8, alpha=0.7, label=r'5$\sigma$')

	axes[0].set_ylabel('CCF')
	axes[1].set_ylabel(r'$\sigma$ of exclusion')
	axes[1].set_xlabel(r'RV shift (km s$^{-1}$)')
	axes[0].legend(fontsize=7, ncol=min(n_damp, 4))
	axes[1].legend(fontsize=8)
	fig.suptitle(
		f'{planet_name} — {mol_label} — {ctx["inst"].upper()} — '
		f'{ob.upper()} — t_exp={t_exp}s — PWV={pwv_med}mm',
		fontsize=10,
	)
	fig.tight_layout()
	fig.savefig(fig_path, dpi=200, bbox_inches='tight')
	print(f'Detectability plot saved to {fig_path}')
	plt.show()
	plt.close(fig)

	return {
		'mol_label'  : mol_label,
		'rv_grid'    : rv_grid,
		'ccf'        : ccf_all,
		'logl'       : logl_all,
		'sigma'      : sigma_all,
		'damp_values': damp_values,
		'det_sigma'  : det_sigma,
	}


def ratri_preview(
	planet_name,
	inst,
	modes,
	ob,
	t_exp,
	planet_model_path,
	qual              = None,
	airmass           = 1.2,
	rv_centre         = 0.0,
	rv_left           = 100.0,
	rv_right          = 100.0,
	damp_values       = None,
	n_damp_auto       = 6,
	sn_mask_threshold = 80,
	noise_seed        = 5,
	rv_inject         = 0.0,
	rv_buffer         = 20.0,
	atm_dir           = './atm_transmission',
	spectra_cache_dir = './spectra_cache',
	star_name         = None,
	library           = None,
	savedir           = './ratri_preview_output',
):
	"""
	Convenience wrapper: single-molecule quick-look tool.
	Calls ratri_preview_setup + ratri_preview_snr + ratri_preview_ccf.
	For multi-molecule runs use ratri_1D.py which calls the setup/ccf
	functions directly to avoid reloading the stellar spectrum each time.
	"""
	ctx = ratri_preview_setup(
		planet_name, inst, modes, ob, t_exp,
		qual=qual, airmass=airmass,
		sn_mask_threshold=sn_mask_threshold,
		atm_dir=atm_dir,
		spectra_cache_dir=spectra_cache_dir,
		star_name=star_name, library=library,
		savedir=savedir,
	)

	ratri_preview_snr(ctx)

	model = load_planet_model(planet_model_path)
	if 'planet_wav' not in model:
		raise ValueError(
			'Cannot auto-extract planet_wav. Use a .txt/.dat file.'
		)
	wav_p    = model['planet_wav'] * 1e4
	data_arr = model['data']
	tr_d      = data_arr[:, 1]
	tr_d_only = data_arr[:, 3] if data_arr.shape[1] >= 4 else data_arr[:, 1]

	import os
	mol_label = os.path.splitext(os.path.basename(planet_model_path))[0]

	result = ratri_preview_ccf(
		ctx, wav_p, tr_d, tr_d_only,
		mol_label=mol_label,
		rv_centre=rv_centre, rv_left=rv_left, rv_right=rv_right,
		rv_inject=rv_inject,
		damp_values=damp_values, n_damp_auto=n_damp_auto,
		noise_seed=noise_seed, rv_buffer=rv_buffer,
	)

	result['wav']    = ctx['wav']
	result['sn_met'] = ctx['sn_met']
	return result
