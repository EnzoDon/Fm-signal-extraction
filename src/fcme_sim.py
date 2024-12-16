import numpy as np
import math
import matplotlib.pyplot as plt
import statistics as st
from sigmf import SigMFFile, sigmffile
from rtlsdr import RtlSdr
import sys
import adi
from scipy.signal import windows, lfilter


#-------------------- Parametros --------------------
sample_rate = 1000  # Frecuencia de muestreo (Hz)
t = np.arange(0, 1, 1/sample_rate)  # Vector de tiempo, 1 segundo
number_of_samples = len(t)  # Número de puntos
center_freq = 250
#print('Number_of_samples: ')
#print(number_of_samples)
SNR_dB = 15  # SNR en dB para la señal de banda ancha
ISR_dB = 20  # ISR en dB para la señal de banda estrecha

# Unidad de frecuencia a graficar
frec_unit = 1    # Hz

#pfa_fcme = 0.0001e-20
#pfa_fcme = 0.01

pfa_fcme = 0.004

#pfa_fcme = 0.009
#pfa_fcme = 0.0002



# Porcentaje de elementos para la muestra inicial Q
percent_elements = 0.01


def calculate_threshold(pfa, sorted_r_signal_array):
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

def run(send_signal):
    #-------------------- Generación de señal de banda ancha (SS BPSK con secuencia de Gold de 63 chips) --------------------

    gold_code = np.random.choice([1, -1], size=63)  # Secuencia de Gold simplificada
    ss_signal = np.tile(gold_code, number_of_samples // len(gold_code) + 1)[:number_of_samples]  # Repetimos para cubrir N
    ss_signal *= np.random.choice([1, -1], size=number_of_samples)  # Modulación BPSK
    ss_signal = ss_signal / np.sqrt(np.mean(ss_signal**2))  # Normalizamos la señal
    #print(np.var(ss_signal))
    #plot_signal_and_spectrum(ss_signal, sample_rate)

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

    #print(np.var(nb_bpsk_signal))
    nb_bpsk_signal = nb_bpsk_signal[:number_of_samples] * np.sqrt(10**(ISR_dB / 10))  # Escalamos para el ISR deseado
    #print(np.var(nb_bpsk_signal))
    #plot_signal_and_spectrum(nb_bpsk_signal,sampling_rate)

    # 4. Generar ruido gaussiano blanco para alcanzar el SNR deseado
    noise_power = np.mean(ss_signal**2) / (10**(SNR_dB / 10))
    noise = np.sqrt(noise_power) * np.random.normal(size=number_of_samples)
    #plot_signal_and_spectrum(noise,fs)

    # Sumar todas las señales para crear la señal compuesta
    sdr_signal = ss_signal + nb_bpsk_signal + noise

    # Sumar todas las señales para crear la señal compuesta
    if(send_signal):
        sdr_signal = ss_signal + nb_bpsk_signal + noise
    else:
        sdr_signal = ss_signal + noise

    # Distancia en frecuencia [Hz] entre las muestras
    dist_of_samples = sample_rate/number_of_samples

    signal_PSD = (np.abs(np.fft.fft(sdr_signal)))**2
    signal_PSD_shifted = np.fft.fftshift(signal_PSD)
    signal_PSD_shifted_normalized = signal_PSD_shifted / (number_of_samples*sample_rate)

    # Rango de frecuencias a analizar
    f_sdr_signal = np.arange(center_freq - (sample_rate/2), center_freq + (sample_rate/2), (sample_rate/number_of_samples))

    # Multiplico cada uno de los valores de la PSD por la distancia entre muestras para obtener la energia de cada muestra
    signal_power_shifted = signal_PSD_shifted_normalized * dist_of_samples
    # Normalizo los valores de energia a partir del maximo valor
    signal_power_shifted_normalized = signal_power_shifted / max(signal_power_shifted)

    # Energia de la señal recibida
    power_of_signal_r = signal_power_shifted_normalized

    #Creamos un arreglo de tuplas. En el primer valor cada tupla tendrá el valor de la energia de la señal,
    # en el segundo valor tendrá la frecuencia al que corresponde ese valor de energia.
    r_signal_array = []
    for i in range(0,len(power_of_signal_r)):
        r_signal_array.append((power_of_signal_r[i],f_sdr_signal[i]))


    # Ordenamos las muestras de manera creciente según su energia.
    sorted_r_signal_array = sorted(r_signal_array, key=lambda x: x[0])
    
    th = calculate_threshold(pfa_fcme,sorted_r_signal_array)


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


    return len(clusters_list_signal)


#print('Cantidad de señales detectadas:')
#print(run())

def generar_arreglo_alternado(cantidad):
    arreglo = [i % 2 for i in range(cantidad)]
    return arreglo

num_iteraciones=10000
array_esperado = generar_arreglo_alternado(num_iteraciones)
array_rtdo = []
for i in range(num_iteraciones):
    array_rtdo.append(run(i%2))

def porcentaje_coincidencia(arreglo1, arreglo2):
    # Verificar que ambos arreglos tengan la misma longitud
    if len(arreglo1) != len(arreglo2):
        raise ValueError("Los arreglos deben tener la misma longitud.")

    coincidencias = sum(1 for a, b in zip(arreglo1, arreglo2) if valid(a,b))
    # Calcular el porcentaje de coincidencia
    porcentaje = (coincidencias / len(arreglo1)) * 100
    return porcentaje

def valid(valor, valorEsperado):
    if(valorEsperado == 0):
        return valor == 0
    else:
        return valor >= valorEsperado


resultado = porcentaje_coincidencia(array_rtdo, array_esperado)
print(f"Porcentaje de coincidencia: {resultado}%")            