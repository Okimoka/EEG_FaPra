# Inital Setup
@Okan list the necessary steps to do the setup for the project (python environments downloads etc)

# Running the Project
@Okan list the necessary steps to run the project (start multiple scripts, go to You:Quantified, enter Streams etc)

# Controls in You:Quantified
* ' ' (space) : pauses/unpauses the demonstration
* 'm' : enables/disables the mirror effect
* '+' : adds one to the symmetry
* '-' : subtracts on of the symmetry
* 'r' : random symmetry

# EEG Stream
The basic ordering of the powerchannels is delta, theta, alpha, beta (which can be changed). The basic order of the EEG channels is the same as in the unicorn device.

The following properties can be adjusted by adjusting the global variables at the top of eeg_streamer.py
* QUANTILE: The quantile used to normalize the powerchannels
* BUFFERSIZE: The size of the ringbuffer for the powerchannels (and the maximal interval for normalization)
* LOWCUT: Lowcut of the bandpassfiltering
* HIGHCUT: Highcut of the bandpassfiltering
* NOTCHFREQ: Frequency of the Notchfiltering
* band_definitions: Frequncyintervals and relevant channels for powerchannels
