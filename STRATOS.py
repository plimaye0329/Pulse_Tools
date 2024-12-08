  1 ## STRATOS : Signal Time-domain Research and Analysis Tools for ObservationS
  2 
  3 # Import all the dependencies: 
  4 #import psrchive
  5 import numpy as np
  6 import matplotlib.pyplot as plt
  7 import glob
  8 from matplotlib.pyplot import figure
  9 from scipy.optimize import curve_fit
 10 from scipy.interpolate import interp1d
 11 import math
 12 from astropy import units as u
 13 from astropy.coordinates import SkyCoord
 14 from astropy.time import Time
 15 from astropy.io import fits
 16 import sigpyproc.readers as readers
 17 import sigpyproc.timeseries as timeseries
 18 import sigpyproc.block as block
 19 from rich.pretty import Pretty
 20 from scipy.signal import find_peaks
 21 from scipy.interpolate import make_interp_spline
 22 import os
 23 import warnings
 24 import math
 25 
 26 ###### PSRCHIVE Implementation ########
 27 
 28 #########Function to plot the dyanamic spectrum from archive file##########
 29 
 30 def plotter(filename, bins=1024):
 31     """
 32     Load a PSRCHIVE archive file, dedisperse it, remove the baseline, and plot the dynamic spectrum and the time series.
 33 
 34     Parameters:
 35     filename (str): The path to the archive file to load.
 36     bins (int, optional): The number of bins to use when generating the time array for plotting the time series. Defaults to 1024.
 37 
 38     Returns:
 39     tuple: A tuple containing the time array (in milliseconds) and the intensity values as a function of time.
 40 
 41     """
 42     import psrchive
 43     archive = psrchive.Archive_load(filename)
 44     #return archive
 45     archive.dedisperse()
 46     archive.remove_baseline()
 47     #archive.remove_chan(150,200)
 48     data = archive.get_data()
 49     #return data
 50     data = data[0][0]
 51     bw = archive.get_bandwidth()
 52     if bw <0:
 53         bw *= -1.
 54     freq_lo = archive.get_centre_frequency()-bw/2.0
 55     freq_hi = archive.get_centre_frequency()+bw/2.0
 56     #return freq_lo,freq_hi
 57 
 58 
 59     #Make timeseries of archive
 60     carch = archive.clone()
 61     carch.fscrunch_to_nchan(1)
 62     ts = carch.get_data()
 63     cropto=False
 64 
 65     if cropto:
 66         ts=ts[:,:,:,cropto[0]:cropto[1]]
 67     ts= ts[0][0][0]
 68 
 69 
 70     #return TS
 71 
 72 
 73 
 74     data[0][0]
 75     fig, ax = plt.subplots(ncols=1,nrows=2,figsize=(8,7),gridspec_kw={
 76                            'height_ratios': [1, 4]})
 77 
 78     ax[0].plot(ts,'k-')
 79     ax[0].set_xlim(0, len(data[0]))
 80 
 81     ax[1].imshow(data,extent=(0,archive.get_nbin(),freq_hi, freq_lo),cmap='viridis',aspect='auto',vmax=5,vmin=-5)
 82 
 83 
 84     plt.tight_layout()
 85 
 86 
 87 #Generating data arrays:
 88     time = np.arange(0,bins,1) #Make x-axis to plot timeseries
 89     it = ts #Intensity Values as a function of time
 90 
 91 #Scaling x-axis to be in millisecond units:
 92     t = (time/bins)*1000 #Time in milliseconds
 93 
 94 #Plotting complete timeseries:
 95     figure(figsize=(7,5),dpi=100)
 96     plt.plot(t,it,label='single pulse data')
 97     plt.xlabel('Time(samples)')
 98     plt.ylabel('Intensity (arb.units)')
 99     plt.legend()
100     plt.show()
101 
102 
103     return data

106 def rfi(data,c1,c2):
107     """
108     Cleans Radio Frequency Interference (RFI) from the provided data.
109 
110     This function identifies bad channels in the data based on the provided range (c1, c2), 
111     flags these bad channels, and then replots the dynamic spectrum with the bad channels flagged.
112 
113     Parameters:
114     data (numpy.ndarray): The input data to be cleaned.
115     c1, c2 (int): The range of bad channels to be flagged.
116 
117     Returns:
118     numpy.ndarray: The cleaned data with bad channels flagged as NaN.
119     """
120     # Step 1: Identify and flag bad channels
121     bad_channels = np.arange(c1,c2,1)  # Replace with the actual bad channels
122 
123     # Step 2: Create a binary mask
124     mask = np.ones_like(data, dtype=bool)
125     mask[bad_channels,:] = False
126 
127     # Step 3: Flag bad channels
128     data[mask == False] = np.nan  # Set bad channels to NaN
129 
130     # Step 4: Replot the dynamic spectrum
131     plt.figure(figsize=(12, 6))
132 
133     plt.subplot(1, 2, 1)
134     plt.imshow(data, cmap='viridis', aspect='auto', vmax=5, vmin=-5)
135     plt.title('Dynamic Spectrum with Bad Channels Flagged')
136 
137     #plt.subplot(1, 2, 2)
138     #plt.imshow(arc_orig, cmap='viridis', aspect='auto', vmax=5, vmin=-5)
139     #plt.title('Original Dynamic Spectrum')
140 
141     plt.tight_layout()
142     plt.show()
143     return data
144 
145 ####Module to scrunch frequencies and plot timeseries####
146 def timeseries(data):
147     """
148     Scrunches the frequencies of the provided data and plots the time series.
149 
150     This function scrunches the frequencies of the data by calculating the mean along the frequency axis. 
151     It then plots the resulting time series.
152 
153     Parameters:
154     data (numpy.ndarray): The input data to be scrunched and plotted.
155 
156     Returns:
157     numpy.ndarray: The frequency-scrunched data.
158     """
159     # Step 4: Scrunch the data in frequency
160     scrunched_data = np.nanmean(data, axis=0)  # You can use np.nansum if you prefer summing
161 
162     # Step 5: Plot the time series
163     plt.figure(figsize=(8, 4))
164 
165     plt.plot(scrunched_data)
166     plt.xlabel('Time (milliseconds)')
167     plt.ylabel('Intensity(arbitrary units)')
168     #plt.xlim(300,980)
169     plt.title('Time Series (Frequency-scrunched)')
170 
171     plt.tight_layout()
172     plt.show()
173     return scrunched_data

###### SIGPYPROC Implementation #######
181 
182 ######### Read Filterbank file and normalize ##########
183 
184 def sigread(filepath, dm, block_size):
185     """
186     Reads a .fil file and returns the dedispersed data.
187 
188     This function reads a .fil file using the FilReader class from the readers module. 
189     It then dedisperses the data using the provided dispersion measure (dm). It then normalizes
190     the data by subtracting the median and dividing by the standard deviation.
191 
192     Parameters:
193     filepath (str): The path to the .fil file to be read.
194     dm (float): The dispersion measure to be used for dedispersion.
195     block_size (int): The size of the block to be read from the .fil file.
196 
197     Returns:
198     numpy.ndarray: The dedispersed data from the .fil file.
199     """
200     fil = readers.FilReader(filepath)
201     header = fil.header
202     data = fil.read_block(1, block_size)
203     data_dedisp = data.dedisperse(dm)
204     arr = data_dedisp.copy()
205     #with warnings.catch_warnings():
206         #warnings.simplefilter("ignore", category=RuntimeWarning)
207         #arr -= np.nanmedian(arr, axis=-1)[..., None]
208         #arr /= np.nanstd(arr, axis=-1)[..., None]
209 
210     return arr
211 
212 
213 ########Clean RFI, plot dynamic spectra and scrunch to timeseries##########
214 def sigclean(data, c1=0, c2 = 0, c3=0, c4=0, c5=0, c6=0, c7=0, c8=0,samp_rate=0.025,filename='cleaned_data.txt'):
215     """
216     Cleans Radio Frequency Interference (RFI) from the provided data, plots the spectrum, and generates a time series.
217 
218     This function identifies bad channels in the data based on the provided ranges (c1 to c8), 
219     flags these bad channels as NaN, and then plots the dynamic spectrum with the bad channels flagged.
220     It also generates a time series by summing the data in frequency, clipping off the first few and last few bits,
221     and then plotting the time series.
222 
223     Parameters:
224     data (numpy.ndarray): The input data to be cleaned.
225     c1, c2, c3, c4, c5, c6, c7, c8 (int): The ranges of bad channels to be flagged.
226     samp_rate: The native sampling rate of the raw data (default=0.025ms)
227     filename (str): The name of the file to save the cleaned data.
228 
229     Returns:
230     numpy.ndarray: The cleaned data with bad channels flagged as NaN.
231     """
232     #Cleaning the RFI channels:
233     mask_channel_ranges = [(c1, c2), (c3,c4), (c5,c6),(c7,c8)]  # replace with your actual mask channel ranges
234 
235     # Mask the channels
236     for start, end in mask_channel_ranges:
237         data[start:end, :] = np.nan
238 
239     #plim=[1,95]
240     #vmin,vmax = np.nanpercentile(data,plim[0]), np.nanpercentile(data,plim[1])
241     #fig = plt.figure()
242     #gs = fig.add_gridspec(2, height_ratios=[0.25, 1], hspace=0.1)
243     #axs = gs.subplots(sharex=True)
244     #axs[1].pcolormesh(data, shading='auto', vmin=vmin, vmax=vmax, rasterized=True)
245     # Scrunch the data in frequency
246     intensity = np.nanmean(data, axis=0)
247 
248     # Clip off the first few and last few bits
249     #intensity = intensity[clip_start:-clip_end]
250 
251     # Generate the time series
252     timeseries = (np.arange(0, len(intensity), 1))*samp_rate
253 
254     # Plot the time series
255     #plt.figure(figsize=(7,5), dpi=100)
256     #plt.plot(timeseries, intensity, label='single pulse data')
257     #plt.xlabel('Time(ms)')
258     #plt.ylabel('Intensity (arb.units)')
259     #plt.show()
260 
261     timeseries_data = np.vstack((timeseries, intensity)).T
262 
263 
264 
265 
266     # Save the cleaned data to a text file
267     np.savetxt(filename, timeseries_data, delimiter=',')

########Function to extract pulses from the timeseries##########
270 
271 #def extract_pulses(ts, it, distance=160, prominence=600, width=200, save_dir='./', filename='B0355'):
272     """
273     Extracts and plots pulses from time series data.
274 
275     Parameters:
276     ts, it (numpy.ndarray): The time series and intensity data.
277     distance (int): The minimum number of samples separating each peak.
278     prominence (int): The required prominence of the peaks.
279     width (int): The number of samples to extract around each peak.
280     save_dir (str): The directory to save the plots.
281     filename (str): The base filename for the plots.
282 
283     Returns:
284     list of numpy.ndarray: The extracted pulses.
285     """
286     # Ensure the save directory exists
287  #   os.makedirs('/hercules/results/pral/Python_Notebooks/B0355_singlepulses', exist_ok=True)
288 
289     # Find the peaks in the intensity data
290   #  peaks, _ = find_peaks(it, distance=distance, prominence=prominence)
291 
292     # Extract a range of samples around each peak
293    # pulses = [it[i-width:i+width] for i in peaks]
294 
295     # Plot each pulse in a separate subplot
296     #for i, (pulse, peak) in enumerate(zip(pulses, peaks), start=1):
297      #   plt.figure(figsize=(10, 4))  # Create a new figure for each plot
298       #  plt.plot(ts[peak-width:peak+width], pulse)
299        # plt.title(f'P{i}')  # Use the counter i for the title
300         #plt.xlabel('Time (samples)')
301         #plt.ylabel('SNR') 
302         #plt.savefig(os.path.join(save_dir, f'{filename}_P{i}.jpg'), format='jpg')  # Save the plot in the specified directory
303         #plt.close()
304 
305     #return pulses
306 
307 
308 ####### Function to eliminate RFI candidates from the list of pulses ########
309 
310 def true_candidates(pulses, indices_to_remove):
311     """
312     Eliminates RFI from a list of pulse candidates.
313 
314     This function removes specific pulses, which are assumed to be RFI, from the pulses list.
315     The pulses to be removed are at the indices specified in indices_to_remove.
316 
317     Parameters:
318     pulses (list of numpy.ndarray): The list of pulse candidates.
319     indices_to_remove (list of int): The indices of the pulses to remove.
320 
321     Returns:
322     list of numpy.ndarray: The list of true pulse candidates, with the specified pulses removed.
323     """
324     true_pulses = [pulse for i, pulse in enumerate(pulses) if i not in indices_to_remove]
325     return true_pulses

############Producing downsampled interpolated pulses############
385 
386 def downsample_profile(profile, factor):
387     if factor <= 0:
388         raise ValueError("Downsampling factor must be greater than zero.")
389     new_length = len(profile) // factor
390     reshaped_profile = profile[:new_length * factor].reshape((new_length, factor))
391     downsampled_profile = np.mean(reshaped_profile, axis=1)
392     return downsampled_profile
393 
394 
395 
396 def interpolate_timeseries(timeseries_file, new_length):
397     """
398     Interpolates a timeseries to a new length and saves the interpolated timeseries to a new text file.
399 
400     This function loads a timeseries from a txt file, performs cubic interpolation to increase its length to new_length,
401     and then saves the interpolated timeseries to a new text file with both time and intensity columns.
402 
403     Parameters:
404     timeseries_file (str): The path to the timeseries txt file.
405     new_length (int): The new length for the interpolated timeseries.
406 
407     Returns:
408     str: The file path for the new interpolated timeseries file.
409     """
410     # Load the timeseries from the txt file
411     timeseries = np.loadtxt(timeseries_file)
412 
413     # Generate the x values (time points)
414     x = np.arange(len(timeseries))
415 
416     # Perform cubic interpolation
417     f = interp1d(x, timeseries, kind='cubic')
418     xnew = np.linspace(0, len(timeseries)-1, new_length)
419     interpolated_timeseries = f(xnew)
420 
421     # Save the interpolated timeseries to a new text file with time and intensity columns
422     interpolated_timeseries_file = f'Interpolated_timeseries.txt'
423     np.savetxt(interpolated_timeseries_file, np.column_stack((xnew, interpolated_timeseries)), header="Time Intensity", fmt="%0.6f",     delimiter="\t")
424 
425     return interpolated_timeseries_file
426 
427 def extract_pulses(ts, it, distance=160, prominence=8, width=200, save_dir='./', filename='B0355'):
428     """
429     Extract pulses from a time series based on peak prominence and distance.
430 
431     Parameters:
432     ts (numpy array): The time series data.
433     it (numpy array): The intensity data.
434     distance (int): Minimum number of samples separating peaks. Default is 160.
435     prominence (int): Required prominence of peaks. Default is 8.
436     width (int): Number of samples to take on either side of the peak. Default is 200.
437     save_dir (str): Directory to save the output files. Default is current directory.
438     filename (str): Base filename for the output files. Default is 'B0355'.
439 
440     Returns:
441     pulses (list): List of numpy arrays representing the extracted pulses.
442     """
443     def interpolate_pulse_data(pulse, new_length):
444         x = np.arange(len(pulse))
445         f = interp1d(x, pulse, kind='cubic')
446         xnew = np.linspace(0, len(pulse)-1, new_length)
447         return f(xnew)
448 
449     os.makedirs(save_dir, exist_ok=True)
450     peaks, _ = find_peaks(it, distance=distance, prominence=prominence)
451     pulses = [it[i-width:i+width] for i in peaks]
452 
453     for i, (pulse, peak) in enumerate(zip(pulses, peaks), start=1):
454         if pulse.size == 0:
455             continue
456         interpolated_pulse = interpolate_pulse_data(pulse, len(pulse))  # Interpolate to the length of the original pulse
457         pulse_data = np.column_stack((np.arange(len(interpolated_pulse)), interpolated_pulse))  # Combine time and intensity
458         np.savetxt(os.path.join(save_dir, f'{filename}_P{i}.txt'), pulse_data, header="Time Intensity", fmt="%0.6f", delimiter="\t")
459 
460     return pulses

######### Function to calculate width and SNR ##########
466 
467 # w50: finds the fwhm of
468 
469 def w_50(pulse_files, resol=0.0002, x_start=None, x_end=None, w50_filename='w50_values.txt', snr_filename='snr_values.txt',w50_err_filename='w50_err.txt'):
470     """
471     pulse_files : Interpolated and extracted single pulse arrays
472     resol : Interpolated time resolution of single pulses (Default : 0.0002seconds)
473     Note: The width values are in units of seconds and not milliseconds
474     """
475     w50_values = []
476     snr_values = []
477     w50_err = []
478 
479     for i, pulse_file in enumerate(pulse_files):
480         pulse = np.loadtxt(pulse_file)
481         x = pulse[...,0]
482         y = pulse[...,1]
483 
484         # Apply the window
485         if x_start is not None and x_end is not None:
486             mask = (x >= x_start) & (x <= x_end)
487             x = x[mask]
488             y = y[mask]
489 
490         # Calculate FWHM
491         ymax = np.max(y)
492         w50 = ymax / 2.
493         crossings = np.where(np.diff(np.sign(y - w50)))[0]
494         if len(crossings) < 2:
495             print(f"Warning: Insufficient data to calculate FWHM for pulse file {pulse_file}")
496             continue
497         left_idx = crossings[0]
498         right_idx = crossings[-1] + 1
499         fwhm = (x[right_idx] - x[left_idx]) * resol
500 
501         w50_values.append(fwhm)
502 
503         # Calculate SNR
504         snr = ymax
505         snr_values.append(snr)
506 
507         fwhm_err = fwhm/snr
508         w50_err.append(fwhm_err)
509 
510         # Plot and save the pulse with shaded FWHM region
511         plt.plot(x, y)
512         plt.axvspan(x[left_idx], x[right_idx], color='gray', alpha=0.5)
513         plt.xlabel('Time (samples)')
514         plt.ylabel('SNR')
515         plt.legend([f'FWHM: {round(fwhm*1000,2)} ms, SNR: {round(snr,2)}'])
516         plt.savefig(f'Pulse_fwhm_{i}.png')
517         plt.close()
518 
519     # Save the FWHM and SNR values to files
520     np.savetxt(w50_filename, w50_values)
521     np.savetxt(snr_filename, snr_values)
522     np.savetxt(w50_err_filename,w50_err)
523 
524 # Get a list of all pulse files
525 #pulse_files = glob.glob('B0355_interp_P*.txt')

# w10: finds the pulse width at 10% of the maximum intensity
528 def w_10(pulse_files, time_resolution=0.025):
529     """
530     Calculates the full width at ten percent maximum (FWTM) and signal-to-noise ratio (SNR) for a list of pulses.
531 
532     This function calculates the FWTM and SNR for each pulse in pulse_files. The FWTM is the width of the pulse at ten
533     percent of its maximum intensity. The SNR is the maximum intensity of the pulse.
534 
535     Each pulse is plotted with the region of the FWTM shaded. The plot is saved as a PNG file with a filename based on
536     the index of the pulse in the list.
537 
538     Parameters:
539     pulse_files (list of str): The list of pulse file paths.
540     time_resolution (float): The time resolution of the pulse data.
541 
542     Returns:
543     tuple: A tuple containing two numpy arrays. The first array contains the FWTM values for each pulse. The second
544     array contains the SNR values for each pulse.
545     """
546     fwtm_values = []
547     snr_values = []
548     for i, pulse_file in enumerate(pulse_files):
549         pulse = np.loadtxt(pulse_file)
550         x = pulse[...,0]
551         y = pulse[...,1]
552         ymax = max(y)
553         ten_percent_max = ymax * 0.1
554         d = np.sign(ten_percent_max - np.array(y[0:-1])) - np.sign(ten_percent_max - np.array(y[1:]))
555         left_idx = np.where(d > 0)[0][0]
556         right_idx = np.where(d < 0)[0][-1]
557         fwtm = (x[right_idx] - x[left_idx])*time_resolution
558         fwtm_values.append(fwtm)
559 
560         snr = ymax
561         snr_values.append(snr)
562 
563         plt.plot(x, y)
564         plt.axvspan(x[left_idx], x[right_idx], color='gray', alpha=0.5)
565         plt.xlabel('Time(samples)')
566         plt.ylabel('SNR')
567         plt.legend([f'FWTM: {fwtm} ms, SNR: {snr}'])
568         plt.savefig(f'Pulse_fwtm_{i}.png')
569         plt.close()
570 
571     # Save the FWTM and SNR values to text files
572     np.savetxt('fwtm_values.txt', fwtm_values)
573     np.savetxt('snr_values.txt', snr_values)
574 
575     return np.array(fwtm_values), np.array(snr_values)
576 
577 
578 ##################### Function to Calculate Pulse Fluence by Integrating over the S/N ###############################
579 
580 def integrated_fluence(pulse_files, constant = 8.5*10**-6, fluence_file = 'integrated_fluences.txt'):
581     """
582     """
583     integrated_fluence = []
584     for i, pulse_file in enumerate(pulse_files):
585         pulse = np.loadtxt(pulse_file)
586         snr = pulse[...,1] # S/N values for each timebin of a pulse profile
587         fluence = np.sum(snr)*constant # Saves the Fluence integrated over pulse profile in Jy-s
588         integrated_fluence.append(fluence)
589 
590     np.savetxt(fluence_file, integrated_fluence)

######## Energy from Integrated Fluence ###################
593 def fluence_to_energy(fluence, l = 3e21, sefd = 17, num_polarizations = 2, delv = 400*1e6):
594     """
595     Calculates the flux, fluence, and energy of each pulse given the pulse width at half maximum (w50) and signal-to-noise ratio (snr).
596 
597     Parameters:
598     w50 (numpy.ndarray): A 1D array of pulse widths at half maximum in seconds.
599     snr (numpy.ndarray): A 1D array of signal-to-noise ratios.
600     l (float, optional): The distance to the source in cm. Default is 3e21.
601     sefd (int, optional): The system equivalent flux density in Jy. Default is 17.
602     num_polarizations (int, optional): The number of polarizations. Default is 2.
603     delv (float, optional): The bandwidth in Hz. Default is 250*1e6.
604 
605     Returns:
606     tuple: A tuple of three 1D numpy arrays. The first array is the flux in Jy, the second array is the fluence in Jy-s, and the third array is the energy in ergs.
607     """
608     #flux = (sefd * snr) / np.sqrt(num_polarizations * delv * w50)  # Calculate Flux in Jy
609     #fluence = flux * w50  # Jy-s
610     #fluence_ms = fluence * 1e3  # Jy-ms
611     energy = fluence * delv * 10e-23 * 4 * np.pi * l**2  # ergs Calculate energy of the pulse
612     return energy
613 
614 
615 
616 ####### Function to calculate the energy of the pulses ######## 
617 
618 def energy(w50, snr, l = 3e21, sefd = 17, num_polarizations = 2, delv = 250*1e6):
619     """
620     Calculates the flux, fluence, and energy of each pulse given the pulse width at half maximum (w50) and signal-to-noise ratio (snr).
621 
622     Parameters:
623     w50 (numpy.ndarray): A 1D array of pulse widths at half maximum in seconds.
624     snr (numpy.ndarray): A 1D array of signal-to-noise ratios.
625     l (float, optional): The distance to the source in cm. Default is 3e21.
626     sefd (int, optional): The system equivalent flux density in Jy. Default is 17.
627     num_polarizations (int, optional): The number of polarizations. Default is 2.
628     delv (float, optional): The bandwidth in Hz. Default is 250*1e6.
629 
630     Returns:
631     tuple: A tuple of three 1D numpy arrays. The first array is the flux in Jy, the second array is the fluence in Jy-s, and the third array is the energy in ergs.
632     """
633     flux = (sefd * snr) / np.sqrt(num_polarizations * delv * w50)  # Calculate Flux in Jy
634     fluence = flux * w50  # Jy-s
635     fluence_ms = fluence * 1e3  # Jy-ms
636     energy = fluence * delv * 10e-23 * 4 * np.pi * l**2  # ergs Calculate energy of the pulse
637     return flux, fluence, energy
638 
639 
640 
641 ######### Energy Distribution with errorbars propagated through width uncertainty ###############
642 
643 def energy_with_uncertainty(w50, w50_err, snr, l=3e21, sefd=17, num_polarizations=2, delv=250*1e6):
644     """
645     Calculates the flux, fluence, and energy of each pulse, including error bars based on the uncertainties in w50.
646 
647     Parameters:
648     w50 (numpy.ndarray): A 1D array of pulse widths at half maximum in seconds.
649     w50_err (numpy.ndarray): A 1D array of uncertainties in w50.
650     snr (numpy.ndarray): A 1D array of signal-to-noise ratios.
651     l (float, optional): The distance to the source in cm. Default is 3e21.
652     sefd (int, optional): The system equivalent flux density in Jy. Default is 17.
653     num_polarizations (int, optional): The number of polarizations. Default is 2.
654     delv (float, optional): The bandwidth in Hz. Default is 250*1e6.
655 
656     Returns:
657     tuple: A tuple containing three 1D numpy arrays: flux (Jy), fluence (Jy-s), energy (ergs), and their respective uncertainties.
658     """
659     # Flux calculation
660     flux = (sefd * snr) / np.sqrt(num_polarizations * delv * w50)
661 
662     # Uncertainty in flux propagation: σ_flux = |∂flux/∂w50| * σ_w50
663     flux_err = (sefd * snr) / (2 * np.sqrt(num_polarizations * delv)) * (-0.5 * w50**(-1.5)) * w50_err
664 
665     # Fluence calculation
666     fluence = flux * w50  # Jy-s
667     fluence_err = np.sqrt((flux_err * w50)**2 + (flux * w50_err)**2)  # Error propagation for fluence
668 
669     # Energy calculation
670     energy = fluence * delv * 10e-23 * 4 * np.pi * l**2  # ergs
671 
672     # Uncertainty in energy propagation: σ_energy = |∂energy/∂fluence| * σ_fluence
673     energy_err = fluence_err * delv * 10e-23 * 4 * np.pi * l**2
674 
675     return flux, fluence, energy, energy_err


680 def plot_energy_distribution(energy_data, num_bins, completeness_limit):
681     """
682     Plots the cumulative energy distribution for a given energy data.
683 
684     Parameters:
685     energy_data (numpy.ndarray): The energy data to plot.
686     num_bins (int): The number of bins to use in the histogram.
687     completeness_limit (float): The energy value above which the data is considered complete.
688 
689     Returns:
690     None
691     """
692     # Calculate the histogram
693     counts, bin_edges = np.histogram(energy_data, bins=num_bins)
694 
695     # Calculate the number of pulses above each energy threshold
696     counts_above_threshold = np.cumsum(counts[::-1])[::-1]
697 
698     # Create a figure with two subplots
699     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
700 
701     # Create a mask where counts_above_threshold is not zero and energy is above the completeness limit
702     mask = (counts_above_threshold != 0) & (bin_edges[1:] > completeness_limit)
703 
704     # Fit a power-law to the cumulative distribution and plot it
705     logx = np.log10(bin_edges[1:][mask])
706     logy = np.log10(counts_above_threshold[mask])
707     #coeffs = np.polyfit(logx, logy, 1)
708     #poly = np.poly1d(coeffs)
709     #yfit = lambda x: 10**poly(np.log10(x))
710     axs[0].loglog(bin_edges[1:], counts_above_threshold,'ro',linestyle='dashed')
711     #axs[0].loglog(bin_edges[1:][mask], yfit(bin_edges[1:][mask]), 'k--', label=f'Power-law index: {coeffs[0]:.2f}')
712     axs[0].axvline(x=completeness_limit, label='Completeness limit', color='black', linestyle='--')
713     axs[0].set_xlabel('Energy')
714     axs[0].set_ylabel('Number of pulses above energy')
715     axs[0].legend()
716     axs[0].set_title('Cumulative Energy Distribution')
717 
718 
719     # Display the plot
720     plt.tight_layout()
721     plt.show()

22 
723 ######### Function to plot energy distribution with uncertainties ##########
724 
725 def modified_energy_distribution(energy_data, energy_errors, num_bins, completeness_limit):
726     """
727     Plots the cumulative energy distribution for a given energy data, including uncertainties from energy errors
728     and Poissonian uncertainties in the distribution of counts.
729 
730     Parameters:
731     energy_data (numpy.ndarray): The energy data to plot.
732     energy_errors (numpy.ndarray): The uncertainties in the energy data.
733     num_bins (int): The number of bins to use in the histogram.
734     completeness_limit (float): The energy value above which the data is considered complete.
735 
736     Returns:
737     None
738     """
739     # Calculate the histogram (ignoring zero bins)
740     counts, bin_edges = np.histogram(energy_data, bins=num_bins)
741 
742     # Calculate the number of pulses above each energy threshold (cumulative distribution)
743     counts_above_threshold = np.cumsum(counts[::-1])[::-1]
744 
745     # Calculate Poissonian uncertainties on the cumulative counts
746     poisson_errors = np.sqrt(counts_above_threshold)
747 
748     # Create a figure with two subplots
749     fig, axs = plt.subplots(1, 2, figsize=(12, 6))
750 
751     # Mask to focus on the part of the distribution that is above the completeness limit
752     mask = (counts_above_threshold != 0) & (bin_edges[1:] > completeness_limit)
753 
754     # Logarithmic values for fitting and plotting
755     logx = np.log10(bin_edges[1:][mask])
756     logy = np.log10(counts_above_threshold[mask])
757 
758     # Plot cumulative energy distribution with Poissonian error bars
759     axs[0].errorbar(bin_edges[1:], counts_above_threshold, yerr=poisson_errors, fmt='ro', linestyle='dashed',
760                     label='Cumulative Counts', capsize=5)
761 
762     # Plot the completeness limit line
763     axs[0].axvline(x=completeness_limit, label='Completeness limit', color='black', linestyle='--')
764 
765     # Set axis labels and title
766     axs[0].set_xlabel('Energy')
767     axs[0].set_ylabel('Number of pulses above energy')
768     axs[0].set_title('Cumulative Energy Distribution')
769     axs[0].legend()
770 
771     # Plot energy distribution with error bars (energy uncertainties)
772     mid_bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
773     axs[1].errorbar(mid_bin_centers, counts, xerr=energy_errors, fmt='bo', linestyle='none', capsize=5,
774                     label='Energy Data with Errors')
775 
776     # Set axis labels and title
777     axs[1].set_xlabel('Energy')
778     axs[1].set_ylabel('Counts')
779     axs[1].set_title('Energy Distribution with Errors')
780     axs[1].legend()
781 
782     # Adjust layout and display the plot
783     plt.tight_layout()
784     plt.show()
