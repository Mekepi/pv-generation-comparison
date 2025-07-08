import numpy as np
from time import perf_counter
from os.path import dirname, abspath, isfile
from pathlib import Path
from psutil import cpu_count
from multiprocessing import Pool
from gzip import open as gzopen
from os import makedirs, listdir
from collections import defaultdict
import pickle
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree
from timeit import timeit

import cProfile
import pstats
import io
from venture_generation import ventures_process

from compiled_venture_functions import city_process, coord_generation

dpq:np.ndarray = ventures_process()['SP']['3502101']['(-20.905187,-51.370022)']

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

""" readt:float = perf_counter()
with open("%s\\data\\ventures\\SP\\[3541406]dg-venture.csv"%(Path(dirname(abspath(__file__))).parent), 'r', 8*1024*1024, encoding='utf-8') as file:
    ventures:list[str] = file.readlines()[1:]

city_timeseries_coords_file:Path = Path("%s\\data\\timeseries_coords\\SP[34538]\\[3541406]_Presidente Prudente_coords.dat"%(Path(dirname(abspath(__file__))).parent))
ctmc:np.ndarray = np.loadtxt(city_timeseries_coords_file, np.float64, delimiter=',', ndmin=2, encoding='utf-8')[:, [1,0]]
#print(perf_counter()-readt) """

def t1() -> dict:
    geocode:str = '3502101'
    coord:str = '(-20.905187,-51.370022)'
    date_power_qty_array:np.ndarray = dpq.copy()
    monolith:bool = True

    if (date_power_qty_array[0,0]>20241231):
        return {}
    
    if monolith:
        timeseries_path:Path =  Path('C:\\Programas\\timeseries\\%s\\monolith\\[%s]timeseries%s.csv.gz'%(states[geocode[:2]], geocode, coord))
    else:
        timeseries_path =  Path(dirname(abspath(__file__))).parent.joinpath('data\\timeseries\\%s\\[%s]\\[%s]timeseries%s.csv.gz'%(states[geocode[:2]], geocode, geocode, coord))
    
    with gzopen(timeseries_path, 'rb', 9) as f:
        text:bytes = f.read(16*1024*1024)
    lines:list[str] = text.decode('utf-8').splitlines()[9:-12]

    min_year:int = min(int(date_power_qty_array[0,0])//10000, 2020)
    i0:int = sum([366 if i%4==0 else 365 for i in range(int(lines[0][:4]), min_year)])*24

    col0:list[str] = []
    col1:list[float] = []
    col2:list[str] = []
    for line in lines[i0:]:
        spline = line.split(',', 6)
        col0.append(spline[0].replace(':', '', 1))
        col1.append(float(spline[1])+float(spline[2])+float(spline[3]))
        col2.append(spline[5])
    #timeseries_array:np.ndarray = np.array([col0, col1, col2], np.float64).T
    year_col:np.ndarray = np.array([col0], np.int64).T#timeseries_array[:, 0 ].astype(np.int64)
    val_cols:np.ndarray = np.array([col1, col2], np.float64).T#timeseries_array[:, 1:].astype(np.float64)


    year_filter:np.ndarray = year_col[:, 0]//(1_0000_0000)
    mask:list[np.ndarray] = [
        year_filter == 2020,
        year_filter == 2021,
        year_filter == 2022,
        year_filter == 2023
    ]

    c59_24:int = 59*24
    c60_24:int = 60*24

    arr2024_vals:np.ndarray  = np.zeros((366*24, 2), np.float64)
    arr2024_vals += val_cols[mask[0]]
    for i in [1, 2, 3]:
        year_vals:np.ndarray = val_cols[mask[i]]
        arr2024_vals[:c59_24] += year_vals[:c59_24]
        arr2024_vals[c60_24:] += year_vals[c59_24:]
        arr2024_vals[c59_24:c60_24] += np.mean([
            year_vals[c59_24-24:c59_24],
            year_vals[c59_24:c60_24]
        ], 0)
    arr2024_vals /= 4

    year_col = np.concatenate((year_col, 2024*(1_0000_0000) + year_col[mask[0]]%(1_0000_0000)), 0)
    val_cols = np.concatenate([val_cols, arr2024_vals], 0)

    
    if (states[geocode[:2]] == 'AC'):
        time_correction:int = 5
    elif (states[geocode[:2]] == 'AM'):
        time_correction = 4
    elif (geocode == '2605459'):
        time_correction = 2
    else:
        time_correction = 3
    
    min_date:np.int64 = np.argwhere(year_col//(1_0000) == date_power_qty_array[0, 0])[0, 0]
    dates:np.ndarray = year_col[min_date:].reshape((year_col.shape[0]-min_date)//24, 24, 1)
    shifted_vals:np.ndarray = np.concatenate([
        val_cols[min_date+time_correction:],
        val_cols[min_date:min_date+time_correction]
    ], 0).reshape((year_col.shape[0]-min_date)//24, 24, 2)
    
    irradiance:np.ndarray = shifted_vals[:, :, 0:1]
    t2m:np.ndarray = shifted_vals[:, :, 1:2]
    

    """ def t_correction(g:np.ndarray, t2m:np.ndarray) -> np.ndarray:
        #Tc:np.ndarray = (t2m+g*25/800)
        #Tc[Tc<25] = 25
        
        #return (1-0.0045*(Tc-25))
        return (1-0.0045*(np.maximum((t2m+g*25/800), 25)-25)) """

    lossless_inst_power:np.ndarray = date_power_qty_array[:, 1].copy()
    date_power_qty_array[:, 1] = (np.round(date_power_qty_array[:, 1].astype(np.float64)*0.95*0.97*0.98, 0)).astype(np.int64)

    lossless_power_array:np.ndarray = np.zeros((irradiance.shape[0], 24, 1), np.int64)
    power_array:np.ndarray = np.zeros((irradiance.shape[0], 24, 1), np.float64)

    js:np.ndarray = np.argwhere(dates[:, 0, 0:1]//(1_0000) == date_power_qty_array[:, 0])[:, 0]
    for i in range(date_power_qty_array[date_power_qty_array[:, 0] < 20250101][:, 0].shape[0]):
        power_array[js[i]:, :] += (date_power_qty_array[i, 1]*np.power(0.995, np.arange(power_array[js[i]:].shape[0])/365.25))[:, np.newaxis, np.newaxis]
        lossless_power_array[js[i]:, :] += lossless_inst_power[i]
    

    Z:np.ndarray = np.concatenate((dates, power_array), 2)
    usedZ:np.ndarray = Z[:, 0, 1]/(10**6)#*1.125
    current_installed_capacity:np.float64 = np.sum(usedZ[-1])

    period = (int(Z[0, 0, 0]//1_0000), int(Z[-1, 0, 0]//1_0000))

    print('ventures:', period, Z[[0,-1], 0,  0].astype(np.str_))

    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d import Axes3D #type: ignore
    from matplotlib.ticker import FixedLocator, FixedFormatter

    x:np.ndarray = np.arange(1, Z.shape[0]+1)
    ax:Axes = plt.axes()
    """ with open('aqui.txt', 'w', 1000, 'utf-8') as fout:
        fout.writelines(['%f,%f\n'%(i, usedZ[i-1]) for i in x]) """
    
    usedZ = usedZ
    x = x
    print(x, usedZ, sep='\n')
    plt.plot(x, usedZ, label='Data')
    

    trendZ:np.ndarray = np.polyfit(np.log10(x[x>=1200]), np.log10(usedZ[x>=1200]), 1)
    print(trendZ)
    plt.plot(np.linspace(1,3500, 4000), (10**trendZ[1])*np.power(np.linspace(1,3500, 4000), trendZ[0]), label='y = %.0e · $t^{%.3f}$'%(10**trendZ[1], trendZ[0]), color='orange')

    """ y:np.ndarray = np.arange(Z.shape[1])

    Y, X = np.meshgrid(y, x)

    Z_max_Y:np.ndarray = np.max(usedZ, axis=1)
    from scipy.ndimage import gaussian_filter1d
    smoothed_Z_max_Y = gaussian_filter1d(Z_max_Y, sigma=10)

    ax:Axes3D = plt.axes(projection='3d')

    ax.plot_surface(X, Y, usedZ, cmap='viridis')
    ax.plot(x, [y.max()]*len(x), smoothed_Z_max_Y, color='#440154', linewidth=2, label='Smoothed Z Max over X')
    ax.view_init(20, -50, 0) """

    all_idxs:list[int] = list(np.unique(Z[:, 0, 0]//1_00_0000, True)[1].astype(int))
    idxs:list[int] = [all_idxs[i] for i in range(0, len(all_idxs), len(all_idxs)//4)]
    month_labels = ['%04i-%02i'%(Z[i, 0, 0]//1_0000_0000, Z[i, 0, 0]%1_0000_0000/1_00_0000) for i in idxs]

    # Set the locations of the ticks to correspond to the first day of each unique month
    x_locator = FixedLocator(idxs)
    ax.xaxis.set_major_locator(x_locator)

    # Set the labels for these ticks using the month strings
    x_formatter = FixedFormatter(month_labels)
    ax.xaxis.set_major_formatter(x_formatter)

    ax.set_xlabel('Data [YYYY-MM]')
    #ax.set_ylabel('Hour of the Day [Hour]')
    ax.set_ylabel('Installed Power [MW]')
    ax.set_title('Ventures PV Yield Across (%02i/%i:%02i/%i)\n[%s]\n\nCurrent Installed Capacity: %.2fMW'%(period[0]%10000//100, period[0]//10000, period[1]%10000//100, period[1]//10000, states[geocode[:2]], current_installed_capacity))
    plt.legend()
    plt.tight_layout()
    plt.show()
    #plt.savefig("%s\\outputs\\Ventures MMD PV Generation\\ventures-%s-(%i_%02i, %i_%02i).png"%(Path(dirname(abspath(__file__))).parent, state_abbreviation, period[0]//10000, period[0]%10000//100,  period[1]//10000, period[1]%10000//100), backend='Agg', dpi=200)
    plt.close()

    return {}

    energy_array:np.ndarray = np.concatenate([
        dates[:, :, :1],
        (irradiance[:, :, :1]*power_array*
            (1-0.0045*(np.maximum((t2m[:, :, :1]+irradiance[:, :, :1]*25/800), 25)-25))
        )/1000,
        irradiance[:, :, :1]*lossless_power_array/1000
    ], 2)
    
    energy_year:np.ndarray = energy_array[:, 0, 0]//(1_0000_0000)
    Z0:dict[int, np.ndarray] = {year:energy_array[energy_year == year] for year in range(int(energy_year[0]), int(energy_year[-1])+1)}
    #print(Z0[2024][302, 6:13])
    
    return Z0

def t2() -> dict:
    geocode:str = '3502101'
    coord:str = '(-20.905187,-51.370022)'
    date_power_qty_array:np.ndarray = dpq.copy()

    if (date_power_qty_array[0,0]>20241231):
        return {}
    
    timeseries_path:Path =  Path('C:\\Programas\\timeseries\\%s\\monolith\\[%s]timeseries%s.csv.gz'%(states[geocode[:2]], geocode, coord))

    min_year:int = min(int(date_power_qty_array[0,0])//10000, 2020)

    col0:list[str] = []
    col1:list[float] = []
    col2:list[str] = []

    with gzopen(timeseries_path, 'rt', 9, 'utf-8') as f:
        [f.readline() for _ in range(9)]

        first_line:str = f.readline()

        i0:int = sum([366 if i%4==0 else 365 for i in range(int(first_line[0][:4]), min_year)])*24

        print(i0)

        [f.readline() for _ in range(i0)]

        print(f.readline())
        return {}
        f.readlines(i0)

        for line in f:
            spline = line.split(',', 6)
            col0.append(spline[0].replace(':', '', 1))
            col1.append(float(spline[1])+float(spline[2])+float(spline[3]))
            col2.append(spline[5])
    #timeseries_array:np.ndarray = np.array([col0, col1, col2], np.float64).T
    year_col:np.ndarray = np.array([col0], np.int64).T#timeseries_array[:, 0 ].astype(np.int64)
    val_cols:np.ndarray = np.array([col1, col2], np.float64).T#timeseries_array[:, 1:].astype(np.float64)


    year_filter:np.ndarray = year_col[:, 0]//(1_0000_0000)
    mask:list[np.ndarray] = [
        year_filter == 2020,
        year_filter == 2021,
        year_filter == 2022,
        year_filter == 2023
    ]

    c59_24:int = 59*24
    c60_24:int = 60*24

    arr2024_vals:np.ndarray  = np.zeros((366*24, 2), np.float64)
    arr2024_vals += val_cols[mask[0]]
    for i in [1, 2, 3]:
        year_vals:np.ndarray = val_cols[mask[i]]
        arr2024_vals[:c59_24] += year_vals[:c59_24]
        arr2024_vals[c60_24:] += year_vals[c59_24:]
        arr2024_vals[c59_24:c60_24] += np.mean([
            year_vals[c59_24-24:c59_24],
            year_vals[c59_24:c60_24]
        ], 0)
    arr2024_vals /= 4

    year_col = np.concatenate((year_col, 2024*(1_0000_0000) + year_col[mask[0]]%(1_0000_0000)), 0)
    val_cols = np.concatenate([val_cols, arr2024_vals], 0)

    
    if (states[geocode[:2]] == 'AC'):
        time_correction:int = 5
    elif (states[geocode[:2]] == 'AM'):
        time_correction = 4
    elif (geocode == '2605459'):
        time_correction = 2
    else:
        time_correction = 3
    
    min_date:np.int64 = np.argwhere(year_col//(1_0000) == date_power_qty_array[0, 0])[0, 0]
    dates:np.ndarray = year_col[min_date:].reshape((year_col.shape[0]-min_date)//24, 24, 1)
    shifted_vals:np.ndarray = np.concatenate([
        val_cols[min_date+time_correction:],
        val_cols[min_date:min_date+time_correction]
    ], 0).reshape((year_col.shape[0]-min_date)//24, 24, 2)
    
    irradiance:np.ndarray = shifted_vals[:, :, 0:1]
    t2m:np.ndarray = shifted_vals[:, :, 1:2]
    #print(np.concatenate((dates, irradiance, t2m), 2)[92, 6:19])
    

    def t_correction(g:np.ndarray, t2m:np.ndarray) -> np.ndarray:
        #Tc:np.ndarray = (t2m+g*25/800)
        #Tc[Tc<25] = 25
        
        #return (1-0.0045*(Tc-25))
        return (1-0.0045*(np.maximum((t2m+g*25/800), 25)-25))

    lossless_inst_power:np.ndarray = date_power_qty_array[:, 1].copy()
    date_power_qty_array[:, 1] = (np.round(date_power_qty_array[:, 1].astype(np.float64)*0.95*0.97*0.98, 0)).astype(np.int64)

    lossless_power_array:np.ndarray = np.zeros((irradiance.shape[0], 24, 1), np.int64)
    power_array:np.ndarray = np.zeros((irradiance.shape[0], 24, 1), np.float64)

    js:np.ndarray = np.argwhere(dates[:, 0, 0:1]//(1_0000) == date_power_qty_array[:, 0])[:, 0]
    for i in range(date_power_qty_array[date_power_qty_array[:, 0] < 20250101][:, 0].shape[0]):
        power_array[js[i]:, :] += (date_power_qty_array[i, 1]*np.power(0.995, np.arange(power_array[js[i]:].shape[0])/365.25))[:, np.newaxis, np.newaxis]
        lossless_power_array[js[i]:, :] += lossless_inst_power[i]

    energy_array:np.ndarray = np.concatenate([
        dates[:, :, :1],
        (irradiance[:, :, :1]*power_array*t_correction(irradiance[:, :, :1], t2m[:, :, :1]))/1000,
        irradiance[:, :, :1]*lossless_power_array/1000
    ], 2)
    
    energy_year:np.ndarray = energy_array[:, 0, 0]//(1_0000_0000)
    Z0:dict[int, np.ndarray] = {year:energy_array[energy_year == year] for year in range(int(energy_year[0]), int(energy_year[-1])+1)}
    
    return Z0

def t3() -> dict:
    geocode:str = '3502101'
    coord:str = '(-20.905187,-51.370022)'
    date_power_qty_array:np.ndarray = dpq.copy()

    Z0:dict[int, np.ndarray] = coord_generation(geocode, coord, date_power_qty_array, True)
    #print(Z0[2024][302, 6:13])

    return Z0

def main() -> None:
    """ lins:list[list[str]] = []
    for line in lines:
        spline:list[str] = line.split(',', 6)
        lins.append([spline[0].replace(':', ''), str(float(spline[1])+float(spline[2])+float(spline[3])), spline[5]])
    cpu_array:np.ndarray = np.array(lins) """
    
    # Tempo médio de execução: 0.2561958704999233 segundos -- for loop já somando os g's e transformado direto em array
    # Máxima leitura paralela das timeseries: ~ 260MB/s
    
    t1()
    #t3()
    return

    """ pr = cProfile.Profile()
    with pr:
        t1()
    
    # Retrieve and print results
    s = io.StringIO()
    pstats.f8 = lambda t: '%8.6f'%(t)
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    with open('analysis_result.txt', 'w', 10124*1024, 'utf-8') as fout:
        fout.write(s.getvalue())
    return """
    
    n:int = 100
    for f in [t1, t3]:
        print(f"Tempo médio de execução: {timeit(f, number=n)/n} segundos")
    
    #print(cpu_array)

if '__main__' == __name__:
    main()