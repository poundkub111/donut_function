import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import fft,fft2,fftn,ifft
from scipy.signal import stft
from scipy import optimize,signal
import OpenRadar_master.mmwave as mm
import copy
import os
from os import listdir
from os.path import isfile,join
from prettytable import PrettyTable

# Basic Function
def pp(x) : print(x)
def ps(x) : print(np.shape(x)) # Check Shape
def pt(x) : print(type(x)) # Check Type
def pl(x) : print(len(x)) # Check length

# Color
BLACK = '\033[90m'
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
PURPLE = '\033[95m'
CYAN = '\033[96m'
GRAY = '\033[97m'

# Style
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

# BackgroundColor
BgBLACK = '\033[40m'
BgRED = '\033[41m'
BgGREEN = '\033[42m'
BgORANGE = '\033[43m'
BgBLUE = '\033[44m'
BgPURPLE = '\033[45m'
BgCYAN = '\033[46m'
BgGRAY = '\033[47m'

# End
END = '\033[0m'

################# List files in folder ###################

def List_file_in_folder(folder_path) :

    # Call all files in folder
    onlyfiles = sorted([f for f in listdir(folder_path) if isfile(join(folder_path,f))])

    # Remove .DS_Store in folder
    [os.remove(os.getcwd()+'/'+folder_path+'/.DS_Store') for i in onlyfiles if i == '.DS_Store']

    # Call all files in folder again
    onlyfiles = sorted([f for f in listdir(folder_path) if isfile(join(folder_path,f))])

    return onlyfiles
    

################# Print selected files Table ###################

def fileTable(raw_data,file_name,file_num) :

    t = PrettyTable(['file number','file name'])
    for i in range(len(file_name)) :
        
        green_color = False
        for j in range(len(file_num)) :
            if i == file_num[j] : green_color = True
            
        if green_color : t.add_row([str(i),GREEN+file_name[i]+END])
        else : t.add_row([str(i),YELLOW+file_name[i]+END])
        
    print(t)

################# Print shape variables Table ###################
    
def shapeTable(var_name_array,var_shape_array,header_name) :
    
    t = PrettyTable([GREEN+header_name+END,'shape'])
    for i in range(len(var_name_array)) :
        t.add_row([var_name_array[i],CYAN+str(var_shape_array[i])+END])
        
    print(t)

############## Radar Configuration ####################

def radarConfig(config) :

    c_value = 3*10**8 #Speed of light in vaccumm
        
    if config == 4 :
        tx_num = 1
        rx_num = 4
        chirp_per_frame = 1
        frame_num = 6000
        frame_per_sec = 1000 #slow_time frequency
        adc_sample = 200 #fast time frequency
        bandwidth = 3.6017*10**9 #Actual Bandwidth

    if config == 10 :
        tx_num = 2
        rx_num = 4
        chirp_per_frame = 8
        frame_num = 6000
        frame_per_sec = 200 #slow_time frequency (1/periodictiy)
        adc_sample = 200 #fast time frequency
        bandwidth = 3.6*10**9 #Actual Bandwidth
        
    if config == 11 :
        tx_num = 2
        rx_num = 4
        chirp_per_frame = 32
        frame_num = 15000
        frame_per_sec = 500 #slow_time frequency (1/periodictiy)
        adc_sample = 64 #fast time frequency
        bandwidth = 3.6017*10**9 #Actual Bandwidth
        

    chirp_per_sec = chirp_per_frame * frame_per_sec
   
    return tx_num,rx_num,chirp_per_sec,chirp_per_frame,frame_per_sec,frame_num,adc_sample,bandwidth,c_value

    
############## New Oraganize function ###################

def newOrganize(raw_data,tx_num,rx_num,adc_sample,frame_num) :
    # for IWR1443BOOST (4 LVDS lane, Complex mode)

    txChirp_num = len(raw_data)//2//rx_num//adc_sample//frame_num
    print(txChirp_num)
    

    ret = np.zeros(len(raw_data)//2 , dtype=complex)

    # Separate IQ data
    ret[0::4] = raw_data[0::8] + 1j * raw_data[4::8]
    ret[1::4] = raw_data[1::8] + 1j * raw_data[5::8]
    ret[2::4] = raw_data[2::8] + 1j * raw_data[6::8]
    ret[3::4] = raw_data[3::8] + 1j * raw_data[7::8]
    
    ret = ret.reshape(frame_num,txChirp_num,adc_sample,rx_num)
    C_txChp_rx_data = ret.transpose((0,1,3,2)) # (30000,16,4,200)
    
    return C_txChp_rx_data,adc_sample #(chirp*tx_num,rx,adc_sample)


################# Split Tx and Chirp from Frame ##################

def splitTxChirp(C_txChp_rx_data,tx_num,chirp_per_sec) :
    
    frame_sub = []
    frame_num = np.shape(C_txChp_rx_data)[0] #30000
    for j in range(frame_num) :
    
        C_tx_chp_rx_data = []
        for i in range(tx_num) :
            
            C_tx_chp_rx_data.append( C_txChp_rx_data[j,i::tx_num] )
            
        #C_tx_chp_rx_data = np.array(C_tx_chp_rx_data)
        frame_sub.append(C_tx_chp_rx_data)
        
        #chirp_num = np.shape(C_tx_chp_rx_data)[1] # Number of Chirp
        
    return np.array(frame_sub) # (30000,2,8,4,200)


'''
####################### Range-FFT ##################### Shared func

def rangeFFT(C_tx_chp_rx_data,chirp_num,rx_num,mode) : #(20000,4,200)

    range_fft = []
    for chirp_loop in range(chirp_num) :
    
        if mode == 'single' :
            range_fft_sub = fft(C_tx_chp_rx_data[0][chirp_loop][rx_num][:]) # Rx 0 (single)
        if mode == 'multi' :
            range_fft_sub = fft(C_tx_chp_rx_data[0][chirp_loop][:rx_num][:]) # 4 Rx (multi)
        
        range_fft.append(range_fft_sub)

    return np.asarray(range_fft)


####################### Static Clutter Removal ###########################

def clutterRemoval(range_fft,adc_sample,chirp_num) :

    range_fft_scr = []
    for j in range(adc_sample) :

        avg_value = complex( np.sum(range_fft[:,j].real,axis=0)/chirp_num , np.sum(range_fft[:,j].imag,axis=0)/chirp_num )
        
        range_fft_scr_sub = []
        for i in range(chirp_num) :

            range_fft_scr_sub.append( range_fft[i,j]-avg_value )
            
        range_fft_scr.append(range_fft_scr_sub)

    return np.asarray(range_fft_scr).T #(20000,200)


####################### Multi Rx Static Clutter Removal ###########################

def mrxClutterRemoval(mrx_range_fft,adc_sample,chirp_num) :

    avg_value = np.sum(mrx_range_fft,axis=0)/chirp_num

    mrx_range_fft_scr = []
    for i in range(chirp_num) :

        mrx_range_fft_scr.append( mrx_range_fft[i,:,:]-avg_value )

    return np.asarray(mrx_range_fft_scr) #(20000,4,200)

'''

####################### Static Clutter Removal ###########################

def clutter_removal_func(range_fft_array) :

    return range_fft_array - range_fft_array.mean(axis=(1,0))

################## Non-wrapping CFAR ######################

def CFAR(range_fft_scr,frame_num) :

    det = []
    for i in range(frame_num) :
       det.append( mm.dsp.caso(np.abs(range_fft_scr[i,:]), l_bound=8000, guard_len=3, noise_len=2, mode='constant') )

    return np.array(det).astype("int")

################## Selected bin calculation #################

def binCalculation(det) :

    data_pt_sum = np.sum(det,axis=0)
    selected_bin = np.argmax(data_pt_sum)
    
    #selected_bin = 24 # bin interrupted!!!

    pp('selected bin')
    pp(selected_bin)
    return selected_bin # (1)


############## Range resolution Calculation #################

def rangeCalculation(selected_bin,adc_sample,bandwidth,c_value) :

    x = np.arange(adc_sample)
    range_bin_res = ( c_value/(2*bandwidth) )*100 # Range resolution in cm
    x[:] = x[:] * range_bin_res
    dist_array = x
    selected_bin_dist = range_bin_res*selected_bin
    
    pp('range bin resolution')
    pp(range_bin_res)
    pp('selected bin distance in cm')
    pp(selected_bin_dist)
    return range_bin_res,selected_bin_dist,dist_array
    
'''############## Range-FFT / Clutter Removal / CFAR plot ##################'''

def range_fft_clutter_removal_cfar_plot(range_fft_array,range_fft_remove,det,adc_sample,chirp_num,range_bin_res,selected_bin_dist,dist_array,selected_bin,show_plot) :

    if show_plot[0] == 1 :

        '''########## Range-FFT plot ###########'''

        x = np.arange(adc_sample)
        y = np.arange(chirp_num)
        x_plot, y_plot = np.meshgrid(x,y)
        z_plot = np.abs(range_fft_array)

        fig = plt.figure(figsize=(14,8))
        ax1 = fig.add_subplot(231,projection='3d')
        ax2 = fig.add_subplot(232)
        ax3 = fig.add_subplot(234,projection='3d')
        ax4 = fig.add_subplot(235)
        ax5 = fig.add_subplot(236)
        ax6 = fig.add_subplot(233)

        ax1.plot_surface(x_plot,y_plot,z_plot,cmap='viridis',edgecolor='none')
        ax1.set_title('3D_range_fft')

        sns.heatmap(z_plot[:,:],cmap='Spectral',ax=ax2)
        ax2.set_xlabel('ADC_sample')
        ax2.set_ylabel('chirp_num')
        ax2.invert_yaxis()
        ax2.set_title('heatmap_range_fft')

        '''############## Static Clutter Removal plot ##################'''

        x = np.arange(adc_sample)
        y = np.arange(chirp_num)
        x_plot, y_plot = np.meshgrid(x,y)

        z_plot = np.abs(range_fft_remove)

        ax3.plot_surface(x_plot,y_plot,z_plot,cmap='viridis',edgecolor='none')
        ax3.set_title('3D_range_fft_clutter_removal')

        sns.heatmap(z_plot[:,:],cmap='Spectral',ax=ax4)
        ax4.set_xlabel('ADC_sample')
        ax4.set_ylabel('chirp_num')
        ax4.invert_yaxis()
        ax4.set_title('heatmap_range_fft_clutter_removal')
        
        '''############### CFAR Plot #################'''

        x = np.arange(adc_sample)
        y = np.arange(chirp_num)
        x_plot, y_plot = np.meshgrid(x,y)

        z_plot = det

        sns.heatmap(z_plot[:,:],cmap='Spectral',ax=ax5)
        ax5.invert_yaxis()
        ax5.set_title('CFAR_range_fft')
        ax5.set_xlabel('ADC_sample')
        
        '''############### Selected bin Plot #################'''
        
        selected_plot = z_plot
        for i in range(chirp_num) :
            selected_plot[i,:] = 0
            selected_plot[i,selected_bin] = 1

        
        xlabels = ['{:3.1f}'.format(xl) for xl in dist_array]
        sns.heatmap(selected_plot[:,:],cmap='Spectral',ax=ax6,xticklabels=xlabels)
        ax6.set_xticks(ax6.get_xticks()[::10])
        ax6.invert_yaxis()
        ax6.set_title('selected_bin = {:3.1f} , selected_distance = {} cm'.format(selected_bin,selected_bin_dist))
        ax6.set_xlabel('Distance in cm')
        plt.tight_layout()
        
        plt.show()
        
        
'''####################### Donut Plot #############################'''

def donut_plot(range_fft_remove,selected_bin,show_plot,limit,x_lim,y_lim) :

    if show_plot[1] == 1 :

        xp = range_fft_remove[:,selected_bin].real
        yp = range_fft_remove[:,selected_bin].imag
        zp = np.arange(len(yp))

        ax = plt.axes(projection='3d')
        ax.scatter3D(xp,yp,zp,c=zp,cmap='Greens')
        ax.plot(xp,yp,zp, color='g')

        if limit :
            plt.xlim(x_lim)
            plt.ylim(y_lim)
                
        plt.title('Selected bin IQ')
        plt.show()

################## DC Offset Correction #########################

def remove_offset_func(range_fft_remove,selected_bin) :

    DCQ = range_fft_remove[:,selected_bin].imag.mean()
    DCI = range_fft_remove[:,selected_bin].real.mean()
    DCC = complex(DCI,DCQ)

    range_fft_remove_offset = range_fft_remove[:,selected_bin]-DCC
    
    return range_fft_remove_offset
    
'''############### DC Offset Correction donunt plot ######################'''

def remove_offset_donut_plot(range_fft_remove_offset,selected_bin,limit,x_lim,y_lim) :

    xp = range_fft_remove_offset.real
    yp = range_fft_remove_offset.imag
    zp = np.arange(len(yp))

    ax = plt.axes(projection='3d')
    ax.scatter3D(xp,yp,zp,c=zp,cmap='Greens')
    ax.plot(xp,yp,zp, color='g')
    
    if limit :
        plt.xlim(x_lim)
        plt.ylim(y_lim)

    plt.title('Selected bin IQ remove offset')
    plt.show()

################### Phase Unwrapping #########################

def phase_unwrap_func(phase) : #phase_arctan_unwrap = phase_unwrap(phase_arctan)

    phase_in_func = copy.deepcopy(phase)

    for n in range(len(phase_in_func)-1) :
        if phase_in_func[n+1] - phase_in_func[n] > np.pi : phase_in_func[n+1:] -= 2*np.pi
        elif phase_in_func[n+1] - phase_in_func[n] < -np.pi : phase_in_func[n+1:] += 2*np.pi
        else : pass

    return phase_in_func


################# Phase difference #################

def phase_diff_func(phase_arctan_unwrap) :

    phase_diff = []

    for i in range(len(phase_arctan_unwrap)-1) :
        phase_diff.append( phase_arctan_unwrap[i+1]-phase_arctan_unwrap[i] )
        
    return phase_diff
    

################# Impulse Noise Removal #################

def impulse_remove_func(phase_diff,phase_arctan_unwrap,impulse_threshold) :

    phase_diff_INR = []

    for i in range(len(phase_diff)-2) :
        forward_diff = phase_diff[i+1]-phase_diff[i+2]
        backward_diff = phase_diff[i+1]-phase_diff[i]
        
        #if i in range(1378,1381):
        #    print(i)
        #    print(forward_diff)
        #    print(backward_diff)
        
        if abs(forward_diff) > impulse_threshold and abs(backward_diff) > impulse_threshold and backward_diff * forward_diff > 0 :
        
            phase_diff_INR.append( (phase_diff[i+2]+phase_diff[i])/2.0 ) # Interpolation
            
        else : phase_diff_INR.append( phase_diff[i+1] )
        
    return phase_diff_INR

#if abs(forward_diff+backward_diff)>impulse_threshold :
    
'''########### Phase arctan / Unwrap / Diff / INR plot ###################'''
    
def phase_arctan_unwrap_diff_INR_plot(phase_arctan,phase_arctan_unwrap,phase_diff,phase_diff_INR,show_plot) :

    if show_plot[2] == 1 :
    
        '''################### Phase Extraction plot #########################'''

        fig = plt.figure(figsize=(14,8))
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)

        ax1.plot(phase_arctan)
        ax1.set_title("phase_arctan")

        '''################### Phase Unwrap plot ###################'''

        ax2.plot(phase_arctan_unwrap)
        ax2.set_title("phase_arctan_unwrap")
        
        '''################# Phase difference plot #################'''

        ax3.plot(phase_diff)
        ax3.set_title('phase_diff')
        
        '''################# Impulse Noise Removal plot #################'''

        ax4.plot(phase_diff_INR)
        ax4.set_title('phase_diff_INR')
        plt.tight_layout()
        plt.show()

        pp('phase_diff_INR')
        ps(phase_diff_INR)













################## Track peak bin ############### Shared func

def trackPeakBin(range_fft_scr,chirp_num,rx_number) :

    track_bin = []
    for j in range(chirp_num) :
    
        if len(range_fft_scr.shape) == 2 :
            track_bin.append(np.argmax(abs(range_fft_scr[j,:])))
        if len(range_fft_scr.shape) == 3 :
            track_bin.append(np.argmax(abs(range_fft_scr[j,rx_number,:])))

    return track_bin # (20000)
    
################## Track peak deg ###############

def trackPeakDeg(mrx_range_fft_scr,track_bin,deg) :

    track_deg = []
    mrx_range_fft_sum = []
    for i in range(len(track_bin)) :
        a = fft(mrx_range_fft_scr[i,:,track_bin[i]],n=deg)
        track_deg.append( np.argmax(abs(a)) )
        mrx_range_fft_sum.append(a)
        
        
    #plt.plot(abs(np.array(mrx_range_fft_sum)[1270]))
    #plt.show()

    return np.array(mrx_range_fft_sum),track_deg # (20000,180) , (20000)
    
################## Track peak ############### Shared func

def trackPeak(mrx_range_fft_sum,track_deg) :

    track_mrx_range_fft = []
    for i in range(len(mrx_range_fft_sum)) :

        track_mrx_range_fft.append(mrx_range_fft_sum[i,track_deg[i]])
 
    return np.array(track_mrx_range_fft)

    
'''############# Track Donut Plot ##############'''
def donutPlot(track_mrx_range_fft,show_plot,limit,x_lim,y_lim) :

    if show_plot[1] == 1 :

        xp = track_mrx_range_fft.real
        yp = track_mrx_range_fft.imag
        zp = np.arange(len(yp))

        ax = plt.axes(projection='3d')
        ax.scatter3D(xp,yp,zp,c=zp,cmap='Greens')
        ax.plot(xp,yp,zp, color='g')

        if limit :
            plt.xlim(x_lim)
            plt.ylim(y_lim)
            
        plt.title('Selected bin IQ')
        plt.show()
        
        
###################### Phase Unwrapping ##################### Shared func

def phaseUnwrap(phase) : #phase_arctan_unwrap = phase_unwrap(phase_arctan)

    phase_in_func = copy.deepcopy(phase)

    for n in range(len(phase_in_func)-1) :
        if phase_in_func[n+1] - phase_in_func[n] > np.pi : phase_in_func[n+1:] -= 2*np.pi
        elif phase_in_func[n+1] - phase_in_func[n] < -np.pi : phase_in_func[n+1:] += 2*np.pi
        else : pass

    return phase_in_func

################### Phase difference ################# Shared func

def phaseDiff(phase_unwrap) :

    phase_diff = []

    for i in range(len(phase_unwrap)-1) :
        phase_diff.append( phase_unwrap[i+1]-phase_unwrap[i] )
        
    return phase_diff
    

################# Impulse Noise Removal ################# Shared func

def INR(phase_diff,impulse_threshold,a,b) :

    phase_diff_INR = copy.deepcopy(phase_diff)
    #phase_diff_INR = []
    for i in range(len(phase_diff_INR)-2) :
    
        forward_diff = phase_diff_INR[i+1]-phase_diff_INR[i+2]
        backward_diff = phase_diff_INR[i+1]-phase_diff_INR[i]
        
        
        if abs(forward_diff) > impulse_threshold and abs(backward_diff) > impulse_threshold and backward_diff * forward_diff > 0 :
            
            #phase_diff_INR.append( (phase_diff[i+2]+phase_diff[i])/2.0 ) # Interpolation
            
            phase_diff_INR[i+1] = (phase_diff_INR[i+2]+phase_diff_INR[i])/2.0
            
            if (a == 1 and b == 1) and (i == 25998 or i == 25999 or i == 26000) :
                print('##################')
                print(i)
                print(phase_diff[i+2])
                print(phase_diff[i])
                print((phase_diff[i+2]+phase_diff[i])/2.0)
            
        #else : phase_diff_INR.append( phase_diff[i+1] )
        #else : phase_diff_INR[i+1] = phase_diff[i+1]

    return phase_diff_INR

'''########## Phase arctan / RO / Unwrap / Diff / INR plot ########## Share func'''
    
def phaseArctanUnwrapDiffINRPlot(phase_arctan,phase_ro,phase_unwrap,phase_diff,phase_diff_INR,show_plot) :

    if show_plot[2] == 1 :

        fig = plt.figure(figsize=(14,8))
        ax1 = fig.add_subplot(511)
        ax2 = fig.add_subplot(512)
        ax3 = fig.add_subplot(513)
        ax4 = fig.add_subplot(514)
        ax5 = fig.add_subplot(515)
        
        '''################### Phase Arctan plot #####################'''

        ax1.plot(phase_arctan)
        ax1.set_title('phase_arctan')
        
        '''################ Phase Remove offset plot ##################'''

        ax2.plot(phase_ro)
        ax2.set_title('phase_ro')

        '''################### Phase Unwrap plot ###################'''

        ax3.plot(phase_unwrap)
        ax3.set_title('phase_unwrap')
    
        
        '''################# Phase difference plot #################'''

        ax4.plot(phase_diff)
        ax4.set_title('phase_diff')
        
        '''################# Impulse Noise Removal plot #################'''

        ax5.plot(phase_diff_INR)
        ax5.set_title('phase_diff_INR')
        
        plt.tight_layout()
        plt.show()

        

'''########## Phase arctan / RO / Unwrap / Diff / INR plot ########## Share func'''
    
def phaseArctanUnwrapDiffINRCompare(phase_arctan,phase_ro,phase_unwrap,phase_diff,phase_diff_INR,track_phase_arctan,track_phase_ro,track_phase_unwrap,track_phase_diff,track_phase_diff_INR,show_plot) :

    if show_plot[2] == 1 :

        fig = plt.figure(figsize=(14,8))
        ax1 = fig.add_subplot(521)
        ax2 = fig.add_subplot(523)
        ax3 = fig.add_subplot(525)
        ax4 = fig.add_subplot(527)
        ax5 = fig.add_subplot(529)
        ax6 = fig.add_subplot(522)
        ax7 = fig.add_subplot(524)
        ax8 = fig.add_subplot(526)
        ax9 = fig.add_subplot(528)
        ax10 = fig.add_subplot(5,2,10)
        
        
        ax1.plot(phase_arctan)
        ax1.set_title('phase_arctan')

        ax2.plot(phase_ro)
        ax2.set_title('phase_ro')

        ax3.plot(phase_unwrap)
        ax3.set_title('phase_unwrap')
    
        ax4.plot(phase_diff)
        ax4.set_title('phase_diff')

        ax5.plot(phase_diff_INR)
        ax5.set_title('phase_diff_INR')
        
        ax6.plot(track_phase_arctan)
        ax6.set_title('track_phase_arctan')

        ax7.plot(track_phase_ro)
        ax7.set_title('track_phase_ro')

        ax8.plot(track_phase_unwrap)
        ax8.set_title('track_phase_unwrap')

        ax9.plot(track_phase_diff)
        ax9.set_title('track_phase_diff')

        ax10.plot(track_phase_diff_INR)
        ax10.set_title('track_phase_diff_INR')
    
        plt.tight_layout()
        plt.show()




######### Band-pass Filter ###########

def BPF(sig,order,filter_type,filter_mode,low_cut,high_cut,chirp_per_sec) :

    if filter_type == 'butterworth' :
        if filter_mode == 'sosfilt' :
            sos1 = signal.butter(order,[low_cut,high_cut],'bp',fs=chirp_per_sec,output='sos')
            waveform = signal.sosfilt(sos1,sig)
            
        if filter_mode == 'filtfilt' :
            b,a = signal.butter(order/2,[low_cut,high_cut],'bp',fs=chirp_per_sec)
            waveform = signal.filtfilt(b,a,sig)
            
    return waveform


################## FFT waveform #####################

def waveformFFT(waveform,window,padding) :

    if window == None : window = 1
    if window == 'hann' : window = signal.windows.hann(len(waveform))
    
    #waveform = np.array(waveform)
    waveform = waveform * window
    waveform_fft = fft(waveform,n=padding)
    
    return waveform_fft


'''################## FFT waveform Plot #####################'''

def waveformFFTPlot(waveform_fft,fs) :

    N = len(waveform_fft)
    xf = np.linspace(0.0,fs//2,N//2)
    
    plt.plot(xf, 2.0/N * np.abs(waveform_fft[:N//2]))














'''
################## find mrx_range_fft_sum_unique ###################

def findMrxRangeFFTSumUnique(mrx_range_fft_scr,track_bin,deg) :

    unique_bin = np.sort(np.unique(track_bin))
    
    #print('track_bin')
    #print(track_bin)
    #pp('unique_bin')
    #pp(unique_bin)
    
    mrx_range_fft_sum_unique = []
    track_deg_unique = []
    for i in unique_bin :
    
        mrx_range_fft_sum_unique_sub = []
        track_deg_unique_sub = []
        for j in range(len(track_bin)) :
        
            a = fft(mrx_range_fft_scr[j,:,i],n=deg)
            mrx_range_fft_sum_unique_sub.append(a)
            track_deg_unique_sub.append( np.argmax(abs(a)) )
            
        mrx_range_fft_sum_unique.append( mrx_range_fft_sum_unique_sub )
        track_deg_unique.append( track_deg_unique_sub )
    
    return np.array(mrx_range_fft_sum_unique) , np.array(track_deg_unique) # (len(unique_bin),20000,180) , (len(unique_bin),20000)
    
################## find track_mrx_range_fft_unique ###################

def findTrackMrxRangeFFTUnique(mrx_range_fft_sum_unique,track_deg_unique) :

    track_mrx_range_fft_unique = []
    for i in range(len(track_deg_unique)) :
        
        track_mrx_range_fft_unique_sub = []
        for j in range(len(track_deg_unique[1])) :

            track_mrx_range_fft_unique_sub.append( mrx_range_fft_sum_unique[i,j,track_deg_unique[i,j]] )
            
        track_mrx_range_fft_unique.append( track_mrx_range_fft_unique_sub )
        
    
    track_mrx_range_fft_unique = np.array(track_mrx_range_fft_unique)
    
    #pp('track_mrx_range_fft_unique')
    #ps(track_mrx_range_fft_unique)
    return track_mrx_range_fft_unique # (len(unique_bin),20000)

################## find track_phase_arctan_unique ###################

def findTrackPhaseArctanUnique(track_mrx_range_fft_unique) :

    track_phase_arctan_unique = []
    for i in range(len(track_mrx_range_fft_unique)) :
    
        track_phase_arctan_unique.append(   np.arctan2(track_mrx_range_fft_unique[i].imag,track_mrx_range_fft_unique[i].real) )

    return np.array(track_phase_arctan_unique)
    
################## Phase Remove Offset #####################

def phaseRemoveOffset(track_phase_arctan,track_bin,track_phase_arctan_unique) :
    
    unique_bin = list(np.sort(np.unique(track_bin)))
    base_ind = unique_bin.index(track_bin[0])
    
    track_phase_ro = []
    for i in range(len(track_bin)) :
    
        ind = unique_bin.index(track_bin[i])
    
        offset = track_phase_arctan_unique[ind,i]-track_phase_arctan_unique[base_ind,i]
        track_phase_ro.append( track_phase_arctan[i] - offset )
    
    return np.array(track_phase_ro)


################## Find most bin in each window length ###################

def findMostBin(track_bin,window_len) :

    window_bin = []
    for i in range(int(len(track_bin)/window_len)) :
    
        sub_bin = track_bin[window_len*i:window_len*(i+1)]
        unique_bin = np.unique(sub_bin)
        current_repeat = 0
         
        for ub in unique_bin:
            count_repeat = sub_bin.count(ub)
            
            if count_repeat > current_repeat :
                current_repeat = count_repeat
                most_repeat = ub
            
        for j in range(window_len) : window_bin.append(most_repeat)
            
    return np.array(window_bin)



#################### Window len Variation ###################

def windowLenVary(raw_data_len,window_len_array,raw_data_array) :

    print('varies window_len')
    window_range_fft_array = []
    window_bin_deg_array = []
    for i in range(raw_data_len) : #file_num = 2

        mrx_range_fft_scr = raw_data_array[i][0]
        track_bin = raw_data_array[i][1]

        window_range_fft_sub = []
        window_bin_deg_sub = []
        for j in range(len(window_len_array)) : # window_len 5

            print(j)
            ################## Find most track bin in each window length #################
            window_bin = findMostBin(track_bin,window_len_array[j])
            ############## Track peak deg ###############
            window_range_fft_sum,window_deg = trackPeakDeg(mrx_range_fft_scr,window_bin,180)
            ################## Find most track deg in each window length #################
            window_deg = findMostBin(window_deg,window_len_array[j]) # window track deg too!
            ################## Track peak ############### Shared func
            window_mrx_range_fft = trackPeak(window_range_fft_sum,window_deg)
            
            window_range_fft_sub.append(window_mrx_range_fft)
            window_bin_deg_sub.append([window_bin,window_deg])
            #plt.plot(window_deg)
            #plt.show()
            
        window_range_fft_array.append(window_range_fft_sub) # (2,5)
        window_bin_deg_array.append(window_bin_deg_sub)
                
        print('################################')
        
    #window_range_fft_array = np.array(window_range_fft_array)
    #ps(window_range_fft_array)

    return window_range_fft_array,window_bin_deg_array



#################### Phase extraction Variation ###################

def phaseVary(raw_data_len,window_array_len,window_range_fft_array) :

    print('varies phase')
    phase_array = []
    for i in range(raw_data_len) : #file_num = 2

        
        phase_array_sub = []
        for j in range(window_array_len) : #window_len = 5
        
            print(j)
            ############### Track Phase Arctan #####################
            phase_arctan =  np.arctan2(window_range_fft_array[i][j].imag,window_range_fft_array[i][j].real)
            ################ Track Phase Unwrapping #####################
            phase_unwrap = phaseUnwrap(phase_arctan)
            ################# Track Phase difference #################
            phase_diff = phaseDiff(phase_unwrap)
            ################# Impulse Noise Removal #################
            phase_diff_INR = INR(phase_diff,0.2,i,j) # INR_tresh = 15

            #phase_array_sub.append( [phase_arctan,phase_unwrap,phase_diff,phase_diff_INR] )
        
            phase_array_sub.append([phase_arctan,phase_unwrap,phase_diff,phase_diff_INR])
            
        phase_array.append(phase_array_sub) # (2,5,4)
        
        print('#######################')


    #phase_array = np.array(phase_array)
    #ps(phase_array)
    
    return phase_array




#################### Waveform Variation ###################

def waveformVary(raw_data_len,window_array_len,phase_array,chirp_per_sec) :

    print('varies waveform')
    waveform_array = []
    for i in range(raw_data_len) : #file_num = 2
        
        waveform_sub = []
        for j in range(window_array_len) : # window_len = 5

            breath_waveform = BPF(phase_array[i][j][3],4,'butterworth','sosfilt',0.1,0.5,chirp_per_sec)
            heart_waveform = BPF(phase_array[i][j][3],4,'butterworth','sosfilt',0.8,6.0,chirp_per_sec)

            waveform_sub.append([breath_waveform,heart_waveform])
                
        waveform_array.append(waveform_sub) # (2,5,2)

        print('#######################')
        

    waveform_array = np.array(waveform_array)
    ps(waveform_array)

    return waveform_array


#################### FFT Variation ###################

def fftVary(raw_data_len,window_array_len,waveform_array,zero_pad) :

    fft_array = []
    for i in range(raw_data_len) : #file_num = 2

        fft_sub = []
        print('varies fft')
        for j in range(window_array_len) : # window_len 5

            print(j)
            breath_fft = waveformFFT(waveform_array[i][j][0],'hann',zero_pad)
            heart_fft = waveformFFT(waveform_array[i][j][1],'hann',zero_pad)

            fft_sub.append([breath_fft,heart_fft])
            
        fft_array.append(fft_sub) #(2,5,2)
            
        print('#########################')

    fft_array = np.array(fft_array)
    ps(fft_array)

    return fft_array





################## Phase plot ###########################


def phasePlot(phase_array,file_len,window_len) :

    row = file_len * window_len
    component = 4 # Phase component = 4

    fig,axs = plt.subplots(nrows=row,ncols=component,figsize=(16,10))
    axs = axs.flatten()
    
    for i in range(file_len) : #file number = 2
    
        for j in range(window_len) : # window_len = 5
    
            for k in range(component) : # Phase component = 4
    
                axs[(i*window_len*component)+(j*component)+k].plot(phase_array[i][j][k])
                axs[(i*window_len*component)+(j*component)+k].set_title("phase_array "+str(i)+str(j)+str(k))

    plt.tight_layout()
    plt.show()
    
    
    
################## Waveform plot ###########################


def waveformPlot(waveform_array,file_len,window_len) :

    row = file_len * window_len
    component = 2 # breath/heart = 2

    fig,axs = plt.subplots(nrows=row,ncols=component,figsize=(16,10))
    axs = axs.flatten()
    
    for i in range(file_len) : #file number = 2
    
        for j in range(window_len) : # window_len = 5
    
            for k in range(component) : # breath/heart = 2

    
                axs[(i*window_len*2)+(j*2)+k].plot(waveform_array[i][j][k])
                axs[(i*window_len*2)+(j*2)+k].set_title("waveform_array "+str(i)+str(j)+str(k))

    plt.tight_layout()
    plt.show()







################## FFT plot ###########################

def fftPlot(fft_array,file_len,window_len,chirp_per_sec) :

    row = file_len * window_len
    component = 2 # breath/heart = 2
    
    N = len(fft_array[0][0][0])
    xf = np.linspace(0.0,chirp_per_sec//2,N//2)

    fig,axs = plt.subplots(nrows=row,ncols=component,figsize=(16,10))
    axs = axs.flatten()
    
    for i in range(file_len) : #file number = 2
    
        for j in range(window_len) : # window_len = 5
    
            for k in range(component) : # breath/heart = 2

                axs[(i*window_len*component)+(j*component)+k].plot(xf,abs(fft_array[i][j][k][:N//2]))
                axs[(i*window_len*component)+(j*component)+k].set_title("fft_array "+str(i)+str(j)+str(k))
                axs[(i*window_len*component)+(j*component)+k].set_xlim([0,6])

    plt.tight_layout()
    plt.show()




################## FFT Merge plot ###########################

def fftMergePlot(fft_array,file_len,window_len_array,chirp_per_sec) :

    window_len = len(window_len_array)
    row = file_len # Merge along window_len
    component = 2 # breath/heart = 2
    
    N = len(fft_array[0][0][0])
    xf = np.linspace(0.0,chirp_per_sec//2,N//2)

    fig,axs = plt.subplots(nrows=row,ncols=component,figsize=(16,10))
    axs = axs.flatten()
    
    for i in range(file_len) : #file number = 2
    
        for j in range(window_len) : # window_len = 5
    
            for k in range(component) : # breath/heart = 2

                axs[(i*component)+k].plot(xf,abs(fft_array[i][j][k][:N//2]),label=str(window_len_array[j]))
                axs[(i*component)+k].set_xlim([0,6])
                
                axs[(i*component)+k].legend(loc='upper right')
                axs[(i*component)+k].set_title("fft_array "+str(i)+str(k))

    plt.tight_layout()
    plt.show()
    
    
    
################ Window Bin + Window Deg Plot #####################
    
def windowBinDegPlot(window_bin_deg_array,file_len,window_len) :

    row = file_len * window_len
    component = 2 # Window bin/Window deg = 2

    fig,axs = plt.subplots(nrows=row,ncols=component,figsize=(16,10))
    axs = axs.flatten()
    
    for i in range(file_len) : #file number = 2
    
        for j in range(window_len) : # window_len = 5
    
            for k in range(component) : #Window bin/Window deg = 2

    
                axs[(i*window_len*2)+(j*2)+k].plot(window_bin_deg_array[i][j][k])
                axs[(i*window_len*2)+(j*2)+k].set_title("Window Bin & Window Deg "+str(i)+str(j)+str(k))

    plt.tight_layout()
    plt.show()
    
    
    

# Apply STFT !!!
'''


