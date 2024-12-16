import numpy as np
import math
import matplotlib.pyplot as plt
import statistics as st
from sigmf import SigMFFile, sigmffile
from rtlsdr import RtlSdr
import sys
import adi
from scipy.signal import windows, lfilter


# Valores posibles para la fuente de muestras
# PlutoSDR
# File
# RTLSDR
# SIM
source_type = "SIM"


#PlutoSDR config
# Frecuencia de muestreo
sample_rate_plutosdr = 3e6              # Hz
# Frecuencia central
center_freq_plutosdr = 100.1e6           # Hz
# Numero de muestras a capturar
number_of_samples_plutosdr = 4096*100
# IP PlutoSDR
ip_number_plutosdr = "ip:192.168.1.36"
# Tipo de ganancia de recepcion
gain_control_mode_plutosdr = 'slow_attack'

# RTLSDR config
# Frecuencia de muestreo
sample_rate_rtlsdr = 3e6            # Hz
# Frecuencia central
center_freq_rtlsdr = 93.1e6         # Hz
freq_correction_rtlsdr = 60         # PPM
gain_rtlsdr = 'auto'
# Numero de muestras a capturar
number_of_samples_rtlsdr = 4096*100



# Archivo con la señal capturada que se va a analizar
path_signal_file = './fm_signal_93_1_4k_samples_notebook_linux.sigmf-data'

# Unidad de frecuencia a graficar
frec_unit = 10**6    # MHz

# Porcentaje de elementos para la muestra inicial Q
percent_elements = 0.01

# Valor en frecuencia [HZ] de la distancia entre dos clusters de señales para considerar que los mismos pertenecen a la misma señal
dist_frec = 400000 # Hz

def plot_signal_and_spectrum(signal, sampling_rate):
    """
    Grafica una señal en el dominio del tiempo y en el dominio de la frecuencia.
    
    :param signal: Array de la señal en el dominio del tiempo.
    :param sampling_rate: Tasa de muestreo de la señal (en Hz).
    """
    
    # Calcular el eje de tiempo
    time = np.arange(len(signal)) / sampling_rate

    # Gráfico en el dominio del tiempo
    plt.figure(figsize=(14, 6))

    # Subplot 1: Señal en el dominio del tiempo
    plt.subplot(1, 2, 1)
    plt.plot(time, signal, color='blue')
    plt.title('Señal en el Dominio del Tiempo')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid(True)

    # Calcular la FFT de la señal
    #fft_spectrum = np.fft.fft(signal, (len(signal)*2))
    fft_spectrum = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_spectrum)
    fft_freq = np.fft.fftfreq(len(signal), 1/sampling_rate)
    
    # Subplot 2: Señal en el dominio de la frecuencia
    plt.subplot(1, 2, 2)
    plt.plot(fft_freq[:len(signal)//2], fft_magnitude[:len(signal)//2], color='green')
    plt.title('Señal en el Dominio de la Frecuencia')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Magnitud')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

sample_rate = 0
center_freq = 0
number_of_samples = 0

if source_type == "PlutoSDR":

   # IP de PlutoSDR
   sdr = adi.Pluto(ip_number_plutosdr)

   # Tipo de ganancia de recepcion
   sdr.gain_control_mode_chan0 = gain_control_mode_plutosdr

   sdr.rx_lo = int(center_freq_plutosdr)
   sdr.sample_rate = int(sample_rate_plutosdr)
   sdr.rx_rf_bandwidth = int(sample_rate_plutosdr)
   sdr.rx_buffer_size = number_of_samples_plutosdr

   # Captura de muestras
   sdr_signal = sdr.rx()/2**14

   sample_rate = sample_rate_plutosdr
   center_freq = center_freq_plutosdr
   number_of_samples = number_of_samples_plutosdr

elif source_type == "File":

   signal_file = sigmffile.fromfile(path_signal_file)
   sdr_signal = signal_file.read_samples().view(np.complex64).flatten()
   sample_rate = signal_file.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
   signal_capture = signal_file.get_capture_info(0)
   center_freq = signal_capture.get(SigMFFile.FREQUENCY_KEY, 0)
   number_of_samples = len(sdr_signal)

elif source_type == "RTLSDR":
    sdr = RtlSdr()
    sdr.sample_rate = sample_rate_rtlsdr
    sdr.center_freq = center_freq_rtlsdr
    sdr.freq_correction = freq_correction_rtlsdr
    sdr.gain = gain_rtlsdr

    sdr_signal = sdr.read_samples(number_of_samples_rtlsdr)

    sdr.close()
    sample_rate = sample_rate_rtlsdr
    center_freq = center_freq_rtlsdr
    number_of_samples = number_of_samples_rtlsdr
elif source_type == "SIM":
    #-------------------- Parametros --------------------
    sample_rate = 1000  # Frecuencia de muestreo (Hz)
    t = np.arange(0, 1, 1/sample_rate)  # Vector de tiempo, 1 segundo
    number_of_samples = len(t)  # Número de puntos
    center_freq = 250
    print('Number_of_samples: ')
    print(number_of_samples)
    SNR_dB = 15  # SNR en dB para la señal de banda ancha
    ISR_dB = 30  # ISR en dB para la señal de banda estrecha

    # Unidad de frecuencia a graficar
    frec_unit = 1    # Hz

    #-------------------- Generación de señal de banda ancha (SS BPSK con secuencia de Gold de 63 chips) --------------------

    gold_code = np.random.choice([1, -1], size=63)  # Secuencia de Gold simplificada
    ss_signal = np.tile(gold_code, number_of_samples // len(gold_code) + 1)[:number_of_samples]  # Repetimos para cubrir N
    ss_signal *= np.random.choice([1, -1], size=number_of_samples)  # Modulación BPSK
    ss_signal = ss_signal / np.sqrt(np.mean(ss_signal**2))  # Normalizamos la señal
    print(np.var(ss_signal))
    plot_signal_and_spectrum(ss_signal, sample_rate)

    #-------------------- Generación de señal de banda estrecha (Sinusoidal) --------------------
    f_sin = 10  # Frecuencia de la señal sinusoidal (Hz)
    nb_sin = np.sqrt(10**(ISR_dB / 10)) * np.sin(2 * np.pi * f_sin * t)  # Amplitud ajustada

    #-------------------- Generación de señal de banda estrecha filtrada (filtrado con root-raised cosine) --------------------
    sps         = 12       # Samples per symbol
    span        = 10      # The filter is truncated to span symbols
    beta        = 0.22      # Excess-bandwidth parameter

    #-------------------- Root Raised Cosine Filter Function --------------------

    def rrcosdesign(beta,span,sps):
        index     = np.arange(-(span*sps)/2,(span*sps)/2+1,1)
        Ts        = sps
        rrcFilter = np.array([])

        for n in index:
            if beta == 0:
                aux       = np.sinc(n/Ts)/np.sqrt(Ts)
                rrcFilter = np.append(rrcFilter,aux)
            else:
                if n == Ts/(4*beta) or n == -Ts/(4*beta):
                    aux       = beta*((np.pi+2)*np.sin(np.pi/(4*beta))+(np.pi-2)*np.cos(np.pi/(4*beta)))/(np.pi*np.sqrt(2*Ts))
                    rrcFilter = np.append(rrcFilter,aux)
                else:
                    a         = np.cos((1+beta)*np.pi*n/Ts)
                    b         = (1-beta)*np.pi*np.sinc((1-beta)*n/Ts)/(4*beta)
                    c         = 4*beta/(np.pi*np.sqrt(Ts))
                    aux       = c*(a+b)/(1-(4*beta*n/Ts)**2)
                    rrcFilter = np.append(rrcFilter,aux)
        return rrcFilter
    rrcFilter = rrcosdesign(beta,span,sps)

    nb_bpsk_signal = np.random.choice([1, -1], size=number_of_samples)  # Señal BPSK
    symbolsUps = np.array([])
    for symbol in nb_bpsk_signal:
        pulse      = np.zeros(sps)
        pulse[0]   = symbol
        symbolsUps = np.concatenate((symbolsUps, pulse))



    nb_bpsk_signal = np.convolve(rrcFilter,symbolsUps)


    n =  np.arange(0, len(nb_bpsk_signal))
    #nb_bpsk_signal = nb_bpsk_signal / np.sqrt(np.mean(nb_bpsk_signal**2))
    nb_bpsk_signal = nb_bpsk_signal / np.sqrt(np.mean(nb_bpsk_signal**2))*np.exp(1j*2*np.pi*250*(n/sample_rate))


    #plot_signal_and_spectrum(nb_bpsk_signal,sampling_rate)

    print(np.var(nb_bpsk_signal))
    nb_bpsk_signal = nb_bpsk_signal[:number_of_samples] * np.sqrt(10**(ISR_dB / 10))  # Escalamos para el ISR deseado
    print(np.var(nb_bpsk_signal))
    #plot_signal_and_spectrum(nb_bpsk_signal,sampling_rate)

    # 4. Generar ruido gaussiano blanco para alcanzar el SNR deseado
    noise_power = np.mean(ss_signal**2) / (10**(SNR_dB / 10))
    noise = np.sqrt(noise_power) * np.random.normal(size=number_of_samples)
    #plot_signal_and_spectrum(noise,fs)

    # Sumar todas las señales para crear la señal compuesta
   # sdr_signal = ss_signal + nb_bpsk_signal + noise
    sdr_signal = ss_signal + noise  
else:
   print("El tipo de fuente de muestras no es correcto. Completar la variable source_type con PlutoSDR o File")
   sys.exit()

# Distancia en frecuencia [Hz] entre las muestras
dist_of_samples = sample_rate/number_of_samples   


if(source_type == 'SIM'):
    signal_PSD = np.fft.fft(sdr_signal)
    signal_PSD_shifted = np.abs(signal_PSD)**2
    signal_PSD_shifted = signal_PSD_shifted[:len(signal_PSD_shifted)//2]
    # Rango de frecuencias a analizar
    f_sdr_signal = np.fft.fftfreq(len(sdr_signal), 1/sample_rate)
    f_sdr_signal = f_sdr_signal[:len(signal_PSD_shifted)]
else:
    signal_PSD = (np.abs(np.fft.fft(sdr_signal)))**2
    signal_PSD_shifted = np.fft.fftshift(signal_PSD)
    # Rango de frecuencias a analizar
    f_sdr_signal = np.arange(center_freq - (sample_rate/2), center_freq + (sample_rate/2), (sample_rate/number_of_samples))


signal_PSD_shifted_normalized = signal_PSD_shifted / (number_of_samples*sample_rate)
signal_PSD_dB = 10.0*np.log10(signal_PSD_shifted_normalized)



fig, signal_psd_dB_plot = plt.subplots()
signal_psd_dB_plot.plot(f_sdr_signal/frec_unit, signal_PSD_dB)
signal_psd_dB_plot.grid()
signal_psd_dB_plot.set(xlabel = 'Frecuencia [MHz]', ylabel = 'Magnitud [dB]')
plt.title('Densidad Espectral de Energia de la señal recibida')
plt.show()

# Multiplico cada uno de los valores de la PSD por la distancia entre muestras para obtener la energia de cada muestra
signal_power_shifted = signal_PSD_shifted_normalized * dist_of_samples
# Normalizo los valores de energia a partir del maximo valor
signal_power_shifted_normalized = signal_power_shifted / max(signal_power_shifted)
signal_power_dB = 10.0*np.log10(signal_power_shifted_normalized)

# Energia de la señal recibida
power_of_signal_r = signal_power_shifted_normalized

fig, signal_power_plot = plt.subplots()
signal_power_plot.plot(f_sdr_signal/frec_unit, power_of_signal_r)
signal_power_plot.grid()
signal_power_plot.set(xlabel = 'Frecuencia [MHz]', ylabel = 'Energia Normalizada')
plt.title('Energia Normalizada de la señal recibida')
plt.show()

fig, signal_power_db_plot = plt.subplots()
signal_power_db_plot.plot(f_sdr_signal/frec_unit, signal_power_dB)
signal_power_db_plot.grid()
signal_power_db_plot.set(xlabel = 'Frecuencia [MHz]', ylabel = 'Energia Normalizada [dB]')
plt.title('Energia Normalizada de la señal recibida en dB')
plt.show()

# Creamos un arreglo de tuplas. En el primer valor cada tupla tendrá el valor de la energia de la señal,
# en el segundo valor tendrá la frecuencia al que corresponde ese valor de energia.
r_signal_array = []
for i in range(0,len(power_of_signal_r)):
  r_signal_array.append((power_of_signal_r[i],f_sdr_signal[i]))


# Ordenamos las muestras de manera creciente según su energia.
sorted_r_signal_array = sorted(r_signal_array, key=lambda x: x[0])

# Indice para graficar las muestras de manera creciente según su energia.
if(source_type == 'SIM'):
    samples_index = np.linspace(0,len(f_sdr_signal),len(f_sdr_signal))
else:
    samples_index = np.linspace(0,(number_of_samples-1),number_of_samples)  


def calculate_threshold(pfa):


    Tcme = -np.log(pfa)

    flag = 1
    # Muestra Inicial Q: toma un procentaje (percent_elements) de los elementos partiendo desde el extremo de menor energia
    initial_samples=sorted_r_signal_array[0:math.ceil((percent_elements*len(sorted_r_signal_array)))]
    threshold = Tcme*st.mean([x[0] for x in initial_samples])
    threshold_old = 0

    tuple_item_threshold_old = []
    for tuple_item in sorted_r_signal_array:
        if tuple_item[0] < threshold_old:
            tuple_item_threshold_old.append(tuple_item)

    len_tuple_item_threshold_old = len(tuple_item_threshold_old)

    while(flag):

        tuple_item_threshold = []
        for tuple_item in sorted_r_signal_array:
            if tuple_item[0] < threshold:
                tuple_item_threshold.append(tuple_item)

        len_tuple_item_threshold = len(tuple_item_threshold)

        if threshold == threshold_old or len_tuple_item_threshold <= len_tuple_item_threshold_old:
            flag = 0
        else:
            len_tuple_item_threshold_old = len_tuple_item_threshold
            threshold_old = threshold
            threshold = Tcme*st.mean([x[0] for x in tuple_item_threshold])

    return threshold


# Valores posibles para el método a utilizar
# lad
# ladWC
# fcme
method = 'lad'
#pfa1_lad = 0.0001e-10
#pfa2_lad = 0.001e-3
pfa1_lad = 0.0002
pfa2_lad = 0.01

pfa_fcme = 0.01
#pfa_fcme = 0.0001e-20


if(method == 'lad'):
    tu = calculate_threshold(pfa1_lad)
    tl = calculate_threshold(pfa2_lad)

    print("Umbral Superior: ")
    print(tu)
    print("Umbral Inferior:")
    print(tl)

    fig, signal_power_sorted_t = plt.subplots()
    signal_power_sorted_t.plot(samples_index, [x[0] for x in sorted_r_signal_array])
    signal_power_sorted_t.axhline(y=tu, color='green', linestyle='--',linewidth=0.5)
    signal_power_sorted_t.axhline(y=tl, color='red', linestyle='--',linewidth=0.5)
    signal_power_sorted_t.grid()
    signal_power_sorted_t.set_yscale('log')
    signal_power_sorted_t.set(ylabel = 'Energia Normalizada')
    plt.title('Energia normalizada ordenada de manera creciente con umbrales del LAD')
    plt.show()

    fig, signal_power_plot_t = plt.subplots()
    signal_power_plot_t.plot(f_sdr_signal/frec_unit, power_of_signal_r)
    signal_power_plot_t.axhline(y=tu, color='green', linestyle='--',linewidth=0.5)
    signal_power_plot_t.axhline(y=tl, color='red', linestyle='--',linewidth=0.5)
    signal_power_plot_t.grid()
    signal_power_plot_t.set(xlabel = 'Frecuencia [MHz]', ylabel = 'Energia Normalizada')
    plt.title('Energia Normalizada de la señal recibida con umbrales del LAD')
    plt.show()

    # Se pasan a dB los valores de los umbrales
    tu_dB = (10.0*np.log10((tu)))
    tl_dB = (10.0*np.log10((tl)))

    fig, signal_power_db_plot_t = plt.subplots()
    signal_power_db_plot_t.plot(f_sdr_signal/frec_unit, signal_power_dB)
    signal_power_db_plot_t.axhline(y=tu_dB, color='green', linestyle='--',linewidth=0.5)
    signal_power_db_plot_t.axhline(y=tl_dB, color='red', linestyle='--',linewidth=0.5)
    signal_power_db_plot_t.grid()
    signal_power_db_plot_t.set(xlabel = 'Frecuencia [MHz]', ylabel = 'Energia Normalizada [dB]')
    plt.title('Energia Normalizada de la señal recibida con umbrales del LAD en dB')
    plt.show()

    cluster_aux = []

    flag_saving_cluster = False

    flag_saving_cluster_first_element = True

    flag_cluster_signal = False

    clusters_list = [] #Guarda lista de clusters sean señal o no. Cada elemento de la lista, es un arreglo de tuplas.

    clusters_list_signal = [] #Guarda lista de clusters que pueden ser señal

    for i in range(0,len(r_signal_array)):

        if r_signal_array[i][0] >= tl:
            flag_saving_cluster = True
            if (flag_saving_cluster_first_element == True):
                if(0 < i):#Consideramos la muestra anterior a cuando la supero lo mismo con la siguiente a cuando empieza a bajar
                    cluster_aux.append(r_signal_array[i-1])
                flag_saving_cluster_first_element = False
            cluster_aux.append(r_signal_array[i])
            if r_signal_array[i][0] >= tu:
                flag_cluster_signal = True

        if((r_signal_array[i][0] < tl and flag_saving_cluster == True) or (i == (len(r_signal_array)-1) and flag_saving_cluster == True)):
            flag_saving_cluster = False
            flag_saving_cluster_first_element = True
            cluster_aux.append(r_signal_array[i])
            clusters_list.append(cluster_aux)
            if flag_cluster_signal == True:
                clusters_list_signal.append(cluster_aux)
            flag_cluster_signal = False
            cluster_aux = []


    fig, signal_list_power_db_no_clus_ady = plt.subplots()
    signal_list_power_db_no_clus_ady.plot(f_sdr_signal/frec_unit, signal_power_dB)
    for i in range(0,len(clusters_list_signal)):
        signal_list_power_db_no_clus_ady.plot([(x[1]/frec_unit) for x in clusters_list_signal[i]], [(10.0*np.log10((y[0]))) for y in clusters_list_signal[i]])
    signal_list_power_db_no_clus_ady.axhline(y=tu_dB, color='green', linestyle='--',linewidth=0.5)
    signal_list_power_db_no_clus_ady.axhline(y=tl_dB, color='red', linestyle='--',linewidth=0.5)
    signal_list_power_db_no_clus_ady.grid()
    signal_list_power_db_no_clus_ady.set(xlabel = 'Frecuencia [MHz]', ylabel = 'Energia Normalizada [dB]')
    plt.title('Energia Normalizada en dB de las señales identificadas mediante metodo LAD sin combinacion de clusters adyacentes')
    plt.show()

    print("Cantidad de Señales detectadas mediante metodo LAD sin combinacion de clusters adyacentes: ")
    print(len(clusters_list_signal))  

elif(method == 'ladWC'):
    # Valor en frecuencia [HZ] de la distancia entre dos clusters de señales para considerar que los mismos pertenecen a la misma señal
    dist_frec = 400000 # Hz

    tu = calculate_threshold(pfa1_lad)
    tl = calculate_threshold(pfa2_lad)

    print("Umbral Superior: ")
    print(tu)
    print("Umbral Inferior:")
    print(tl)

    fig, signal_power_sorted_t = plt.subplots()
    signal_power_sorted_t.plot(samples_index, [x[0] for x in sorted_r_signal_array])
    signal_power_sorted_t.axhline(y=tu, color='green', linestyle='--',linewidth=0.5)
    signal_power_sorted_t.axhline(y=tl, color='red', linestyle='--',linewidth=0.5)
    signal_power_sorted_t.grid()
    signal_power_sorted_t.set_yscale('log')
    signal_power_sorted_t.set(ylabel = 'Energia Normalizada')
    plt.title('Energia normalizada ordenada de manera creciente con umbrales del LAD')
    plt.show()

    fig, signal_power_plot_t = plt.subplots()
    signal_power_plot_t.plot(f_sdr_signal/frec_unit, power_of_signal_r)
    signal_power_plot_t.axhline(y=tu, color='green', linestyle='--',linewidth=0.5)
    signal_power_plot_t.axhline(y=tl, color='red', linestyle='--',linewidth=0.5)
    signal_power_plot_t.grid()
    signal_power_plot_t.set(xlabel = 'Frecuencia [MHz]', ylabel = 'Energia Normalizada')
    plt.title('Energia Normalizada de la señal recibida con umbrales del LAD')
    plt.show()

    # Se pasan a dB los valores de los umbrales
    tu_dB = (10.0*np.log10((tu)))
    tl_dB = (10.0*np.log10((tl)))

    fig, signal_power_db_plot_t = plt.subplots()
    signal_power_db_plot_t.plot(f_sdr_signal/frec_unit, signal_power_dB)
    signal_power_db_plot_t.axhline(y=tu_dB, color='green', linestyle='--',linewidth=0.5)
    signal_power_db_plot_t.axhline(y=tl_dB, color='red', linestyle='--',linewidth=0.5)
    signal_power_db_plot_t.grid()
    signal_power_db_plot_t.set(xlabel = 'Frecuencia [MHz]', ylabel = 'Energia Normalizada [dB]')
    plt.title('Energia Normalizada de la señal recibida con umbrales del LAD en dB')
    plt.show()

    cluster_aux = []

    flag_saving_cluster = False

    flag_saving_cluster_first_element = True

    flag_cluster_signal = False

    clusters_list = [] #Guarda lista de clusters sean señal o no. Cada elemento de la lista, es un arreglo de tuplas.

    clusters_list_signal = [] #Guarda lista de clusters que pueden ser señal

    for i in range(0,len(r_signal_array)):

        if r_signal_array[i][0] >= tl:
            flag_saving_cluster = True
            if (flag_saving_cluster_first_element == True):
                if(0 < i):#Consideramos la muestra anterior a cuando la supero lo mismo con la siguiente a cuando empieza a bajar
                    cluster_aux.append(r_signal_array[i-1])
                flag_saving_cluster_first_element = False
            cluster_aux.append(r_signal_array[i])
            if r_signal_array[i][0] >= tu:
                flag_cluster_signal = True

        if((r_signal_array[i][0] < tl and flag_saving_cluster == True) or (i == (len(r_signal_array)-1) and flag_saving_cluster == True)):
            flag_saving_cluster = False
            flag_saving_cluster_first_element = True
            cluster_aux.append(r_signal_array[i])
            clusters_list.append(cluster_aux)
            if flag_cluster_signal == True:
                clusters_list_signal.append(cluster_aux)
            flag_cluster_signal = False
            cluster_aux = []


    fig, signal_list_power_db_no_clus_ady = plt.subplots()
    signal_list_power_db_no_clus_ady.plot(f_sdr_signal/frec_unit, signal_power_dB)
    for i in range(0,len(clusters_list_signal)):
        signal_list_power_db_no_clus_ady.plot([(x[1]/frec_unit) for x in clusters_list_signal[i]], [(10.0*np.log10((y[0]))) for y in clusters_list_signal[i]])
    signal_list_power_db_no_clus_ady.axhline(y=tu_dB, color='green', linestyle='--',linewidth=0.5)
    signal_list_power_db_no_clus_ady.axhline(y=tl_dB, color='red', linestyle='--',linewidth=0.5)
    signal_list_power_db_no_clus_ady.grid()
    signal_list_power_db_no_clus_ady.set(xlabel = 'Frecuencia [MHz]', ylabel = 'Energia Normalizada [dB]')
    plt.title('Energia Normalizada en dB de las señales identificadas mediante metodo LAD sin combinacion de clusters adyacentes')
    plt.show()

    print("Cantidad de Señales detectadas mediante metodo LAD sin combinacion de clusters adyacentes: ")
    print(len(clusters_list_signal))
    signal_aux = []

    signal_list = []

    if len(clusters_list_signal) != 0:
        # Tomo el primer cluster y lo agrego como la primer señal

        for i in range(0,len(clusters_list_signal[0])):
            signal_aux.append(clusters_list_signal[0][i])

        signal_list.append(signal_aux)

        signal_aux = []

        # Calculo el numero de muestras de distancia entre dos clusters para considerar que los mismos pertenecen a la misma señal
        # a partir de la distancia en frecuencia especificada

        num_samples = math.ceil(dist_frec/dist_of_samples)

        # Luego empiezo a comparar la ultima señal agregada a signal_list con los clusters siguientes
        # Por eso comienza en 1

        for i in range(1,len(clusters_list_signal)):

            index_last_signal = len(signal_list)-1 #Indice para acceder al ultimo cluster agregado
            index_last_sample_last_signal = len(signal_list[index_last_signal])-1 #Indice de la ultima posicion del ultimo cluster agregado
            sample_id_last_sample_last_signal = signal_list[index_last_signal][index_last_sample_last_signal][1] #Posicion de frecuencia, de la ultima muestra del ultimo cluster agregado

            sample_id_first_sample_current_cluster = clusters_list_signal[i][0][1] #Posicion de frecuencia del cluster que se esta evaluando

            for j in range(0,len(clusters_list_signal[i])):
                signal_aux.append(clusters_list_signal[i][j])

            if((sample_id_first_sample_current_cluster - sample_id_last_sample_last_signal) <= num_samples): #Si la diferencia entre la ultima posicion de frecuencia del ultimo cluster agregado con la primera es menor o igual que num_samples entonces la considero como la misma señal

                index_start_signal_aux = 0
                if((sample_id_first_sample_current_cluster - sample_id_last_sample_last_signal) == 0):
                    index_start_signal_aux = 1

                for j in range(index_start_signal_aux,len(signal_aux)):
                    signal_list[index_last_signal].append(signal_aux[j]) #Termino metiendo todas las tuplas del cluster que estas analizando en el ultimo cluster metido

            else:
                signal_list.append(signal_aux) #Si no se cumplio lo agregas como un cluster porque es uno más

            signal_aux = []

    fig, signal_list_power_db = plt.subplots()
    signal_list_power_db.plot(f_sdr_signal/frec_unit, signal_power_dB)
    for i in range(0,len(signal_list)):
        signal_list_power_db.plot([(x[1]/frec_unit) for x in signal_list[i]], [(10.0*np.log10((y[0]))) for y in signal_list[i]])
    signal_list_power_db.axhline(y=tu_dB, color='green', linestyle='--',linewidth=0.5)
    signal_list_power_db.axhline(y=tl_dB, color='red', linestyle='--',linewidth=0.5)
    signal_list_power_db.grid()
    signal_list_power_db.set(xlabel = 'Frecuencia [MHz]', ylabel = 'Energia Normalizada [dB]')
    plt.title('Energia Normalizada en dB de las señales identificadas mediante metodo LAD con combinacion de clusters adyacentes')
    plt.show()   

    fig, signal_list_power = plt.subplots()
    signal_list_power.plot(f_sdr_signal/frec_unit, power_of_signal_r)
    for i in range(0,len(signal_list)):
        signal_list_power.plot([(x[1]/frec_unit) for x in signal_list[i]], [y[0] for y in signal_list[i]])
    signal_list_power.axhline(y=tu, color='green', linestyle='--',linewidth=0.5)
    signal_list_power.axhline(y=tl, color='red', linestyle='--',linewidth=0.5)
    signal_list_power.grid()
    signal_list_power.set(xlabel = 'Frecuencia [MHz]', ylabel = 'Energia Normalizada')
    plt.title('Energia Normalizada de las señales identificadas mediante metodo LAD con combinacion de clusters adyacentes')
    plt.show()

    print("Cantidad de Señales detectadas mediante metodo LAD con combinacion de clusters adyacentes: ")
    quantitySignals = len(signal_list)
    print(quantitySignals) 
elif(method == 'fcme'):
    th = calculate_threshold(pfa_fcme)

    print("Umbral:")
    print(th)

    fig, signal_power_sorted_t = plt.subplots()
    signal_power_sorted_t.plot(samples_index, [x[0] for x in sorted_r_signal_array])
    signal_power_sorted_t.axhline(y=th, color='green', linestyle='--',linewidth=0.5)
    signal_power_sorted_t.grid()
    signal_power_sorted_t.set_yscale('log')
    signal_power_sorted_t.set(ylabel = 'Energia Normalizada')
    plt.title('Energia normalizada ordenada de manera creciente con umbral FCME')
    plt.show()

    fig, signal_power_plot_t = plt.subplots()
    signal_power_plot_t.plot(f_sdr_signal/frec_unit, power_of_signal_r)
    signal_power_plot_t.axhline(y=th, color='green', linestyle='--',linewidth=0.5)
    signal_power_plot_t.grid()
    signal_power_plot_t.set(xlabel = 'Frecuencia [MHz]', ylabel = 'Energia Normalizada')
    plt.title('Energia Normalizada de la señal recibida con umbral FCME')
    plt.show()

    # Se pasan a dB el valor del umbral
    th_dB = (10.0*np.log10((th)))

    fig, signal_power_db_plot_t = plt.subplots()
    signal_power_db_plot_t.plot(f_sdr_signal/frec_unit, signal_power_dB)
    signal_power_db_plot_t.axhline(y=th_dB, color='green', linestyle='--',linewidth=0.5)
    signal_power_db_plot_t.grid()
    signal_power_db_plot_t.set(xlabel = 'Frecuencia [MHz]', ylabel = 'Energia Normalizada [dB]')
    plt.title('Energia Normalizada de la señal recibida con umbral FCME en dB')
    plt.show()

    cluster_aux = []

    flag_saving_cluster = False

    flag_saving_cluster_first_element = True

    flag_cluster_signal = False

    clusters_list = [] #Guarda lista de clusters.

    clusters_list_signal = [] #Guarda lista de clusters que pueden ser señal

    for i in range(0,len(r_signal_array)):

        if r_signal_array[i][0] >= th:
            flag_saving_cluster = True
            if (flag_saving_cluster_first_element == True):
                if(0 < i):#Consideramos la muestra anterior a cuando la supero lo mismo con la siguiente a cuando empieza a bajar
                    cluster_aux.append(r_signal_array[i-1])
                flag_saving_cluster_first_element = False
            cluster_aux.append(r_signal_array[i])
            flag_cluster_signal = True

        if((r_signal_array[i][0] < th and flag_saving_cluster == True) or (i == (len(r_signal_array)-1) and flag_saving_cluster == True)):
            flag_saving_cluster = False
            flag_saving_cluster_first_element = True
            cluster_aux.append(r_signal_array[i])
            clusters_list.append(cluster_aux)
            if flag_cluster_signal == True:
                clusters_list_signal.append(cluster_aux)
            flag_cluster_signal = False
            cluster_aux = []


    fig, signal_list_power_db_no_clus_ady = plt.subplots()
    signal_list_power_db_no_clus_ady.plot(f_sdr_signal/frec_unit, signal_power_dB)
    for i in range(0,len(clusters_list_signal)):
        signal_list_power_db_no_clus_ady.plot([(x[1]/frec_unit) for x in clusters_list_signal[i]], [(10.0*np.log10((y[0]))) for y in clusters_list_signal[i]])
    signal_list_power_db_no_clus_ady.axhline(y=th_dB, color='green', linestyle='--',linewidth=0.5)
    signal_list_power_db_no_clus_ady.grid()
    signal_list_power_db_no_clus_ady.set(xlabel = 'Frecuencia [MHz]', ylabel = 'Energia Normalizada [dB]')
    plt.title('Energia Normalizada en dB de las señales identificadas mediante metodo FCME')
    plt.show()

    print("Cantidad de Señales detectadas mediante metodo FCME: ")
    quantitySignals = len(clusters_list_signal)
    print(quantitySignals)
else:
    print("Método incorrecto") 
