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
from timeit import timeit
from geopandas import read_file
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
from datetime import datetime


from usine_generation import usine_plot, usines_plot, usines_pv_mmd_generation
from venture_generation import ventures_process, save_ventures_timeseries_coords_filtered, plot_generation
from compiled_venture_functions import coord_generation

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

def read_gdbtable() -> None:
    for folder_path in [file_path for file_path in listdir('data') if file_path.startswith('Enel_SP') and (not file_path.endswith('.zip'))]:

        file_path:str = '%s\\data\\%s\\%s\\a000000%i.gdbtable'%(Path(dirname(abspath(__file__))).parent, folder_path, folder_path, 50)
        gdbin:pd.DataFrame = pd.DataFrame(read_file(file_path))
        columns_names:list[str] = list(gdbin.columns)[:3] + ['POT_%02i'%(i) for i in range(0, 24)] + list(gdbin.columns)[99:]

        gdbin = pd.DataFrame(
            np.concatenate([
                gdbin.iloc[:, :3],
                pd.DataFrame(np.round(np.sum(gdbin.iloc[:, 3:99].values.reshape((-1, 24, 4)).astype(np.float64), 2)/4, 3)),
                gdbin.iloc[:, 99:]], axis=1),
            columns=columns_names).sort_values([columns_names[0], columns_names[2]])
        
        gdbin.to_csv('data\\avg_load_curve_by_class.csv', index=False)

        for i in [61, 62, 63]:
            file_path = '%s\\data\\%s\\%s\\a000000%i.gdbtable'%(Path(dirname(abspath(__file__))).parent, folder_path, folder_path, i)
            
            row_slice:int = 3000000
            while (row_slice > 0):
                gdbslice:pd.DataFrame = pd.DataFrame(read_file(file_path, rows=slice(row_slice-3000000, row_slice)))

                if row_slice == 3000000 and i == 61:
                    gdbin = gdbslice[gdbslice['CEG_GD'] != ' ']
                else:
                    gdbin = pd.concat([gdbin, gdbslice[gdbslice['CEG_GD'] != ' ']])
                
                print(gdbin.shape, row_slice, i)
                row_slice = row_slice+3000000 if (gdbslice.shape[0] == 3000000) else 0

        print(gdbin.shape)
        gdbin.to_csv('data\\uc_all_types.csv', index=False)

    """ print(gdbin.iloc[:3, 3:27].values)
    for i in range(3):
        plt.plot(range(24), gdbin.iloc[i, 3:27].values)
    plt.show() """

def venture_gen(geocode:str, coordN:str, coordE:str, power:str, initial_date:str) -> dict[int, np.ndarray]:
    
    timeseries_coords_path:Path = Path(dirname(abspath(__file__))).parent.joinpath('data\\timeseries_coords')
    state_timeseries_coords_path:str = '%s\\%s'%(
        timeseries_coords_path,
        [folder for folder in listdir(timeseries_coords_path) if folder.startswith(states[geocode[:2]])][0]
    )
    city_timeseries_coords_path:str = '%s\\%s'%(
        state_timeseries_coords_path,
        [file for file in listdir(state_timeseries_coords_path) if file.startswith('[%s]'%(geocode))][0]
    )

    with open(city_timeseries_coords_path, 'r', 1024*1024*16, 'utf-8') as fin:
        coords:np.ndarray = np.array([[line[:-1].split(',')[1], line[:-1].split(',')[0]] for line in fin.readlines()], dtype=np.float64)
    
    closest_coord:np.ndarray = coords[KDTree(coords).query([float(coordN), float(coordE)], 1)[1]]

    timeseries_path =  Path(dirname(abspath(__file__))).parent.joinpath('data\\timeseries\\%s\\[%s]timeseries(%.6f,%.6f).csv.gz'%(states[geocode[:2]], geocode, *closest_coord))
    
    with gzopen(timeseries_path, 'rb', 9) as f:
        text:bytes = f.read(16*1024*1024)
    lines:list[str] = text.decode('utf-8').splitlines()[9:-12]

    min_year:int = min(int(initial_date)//10000, 2020)
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
    
    min_date:np.int64 = np.argwhere(year_col//(1_0000) == int(initial_date))[0, 0]
    dates:np.ndarray = year_col[min_date:].reshape((year_col.shape[0]-min_date)//24, 24, 1)
    shifted_vals:np.ndarray = np.concatenate([
        val_cols[min_date+time_correction:],
        val_cols[min_date:min_date+time_correction]
    ], 0).reshape((year_col.shape[0]-min_date)//24, 24, 2)
    
    irradiance:np.ndarray = shifted_vals[:, :, 0:1]
    t2m:np.ndarray = shifted_vals[:, :, 1:2]
    

    def t_correction(g:np.ndarray, t2m:np.ndarray) -> np.ndarray:
        #Tc:np.ndarray = (t2m+g*25/800)
        #Tc[Tc<25] = 25
        
        #return (1-0.0045*(Tc-25))
        return (1-0.0045*(np.maximum((t2m+g*25/800), 25)-25))

    #lossless_inst_power:np.ndarray = date_power_qty_array[:, 1].copy()
    power_corrected:np.float64 = np.float64(power)*0.95*0.97*0.98

    #lossless_power_array:np.ndarray = np.zeros((irradiance.shape[0], 24, 1), np.int64)
    power_array:np.ndarray = np.zeros((irradiance.shape[0], 24, 1), np.float64)

    #print(dates)

    js:np.int32 = np.argwhere(dates[:, 0, 0]//(1_0000) == int(initial_date))[0, 0]
    power_array[js:, :] = (power_corrected*np.power(0.995, np.arange(power_array[js:].shape[0])/365.25))[:, np.newaxis, np.newaxis]

    energy_array:np.ndarray = np.concatenate([
        dates[:, :, :1],
        (irradiance[:, :, :1]*power_array*t_correction(irradiance[:, :, :1], t2m[:, :, :1]))
    ], 2)
    
    energy_year:np.ndarray = energy_array[:, 0, 0]//(1_0000_0000)
    Z0:dict[int, np.ndarray] = {year:energy_array[energy_year == year] for year in range(int(energy_year[0]), int(energy_year[-1])+1)}
    
    return Z0

def disaggregation_curve() -> None:
    with open('%s\\data\\avg_load_curve_by_class.csv'%(Path(dirname(abspath(__file__))).parent), 'r', 1024*1024, 'utf-8') as fin:
        columns:list[str] = fin.readline()[:-1].split(',')
        curve_per_class:pd.DataFrame = pd.DataFrame([line[:-1].split(',') for line in fin.readlines()], columns=columns)
    
    #print(curve_per_class)

    with open('%s\\data\\ucbt.csv'%(Path(dirname(abspath(__file__))).parent), 'r', 1024*1024*256, 'utf-8') as fin:
        columns = fin.readline()[:-1].split(',')
        sample1:pd.DataFrame = pd.DataFrame([line[:-1].split(',') for line in fin.readlines()], columns=columns).iloc[:, [0, 15, *range(26, 38), 9, 10]]

    #print(sample1.groupby('MUN').size().sort_values(ascending=False))
    geocode:str = '3550308'
    sample1 = sample1[sample1['MUN'] == geocode]

    #print(sample1['TIP_CC'].unique().size)
    
    #class_samples1:pd.DataFrame = sample1.groupby('TIP_CC').sample(1)
    base_sample:list[str] = [
        'GD.SP.000.289.091', 
        'GD.SP.000.695.545', 
        'GD.SP.000.822.928', 
        'GD.SP.000.863.429', 
        'GD.SP.000.970.959', 
        'GD.SP.001.273.872', 
        'GD.SP.001.373.672', 
        'GD.SP.001.373.967', 
        'GD.SP.001.374.652', 
        'GD.SP.001.374.801', 
        'GD.SP.001.567.071', 
        'GD.SP.001.819.077', 
        'GD.SP.001.932.722', 
        'GD.SP.001.964.944', 
        'GD.SP.002.040.595', 
        'GD.SP.002.575.032', 
        'GD.SP.002.747.040'
    ]
    
    class_samples1:pd.DataFrame = sample1[sample1['CEG_GD'].isin(base_sample)]
    #print(class_samples1)

    with open('%s\\data\\ventures\\SP\\[%s]dg-venture.csv'%(Path(dirname(abspath(__file__))).parent, geocode), 'r', 1024*1024*8, 'utf-8') as fin:
        columns = fin.readline()[1:-2].split('";"')
        city_ventures:pd.DataFrame = pd.DataFrame([line[1:-2].split('";"') for line in fin.readlines()], columns=columns)
    
    
    city_ventures = (city_ventures[city_ventures['CodEmpreendimento'].isin(class_samples1['CEG_GD'])]).sort_values('CodEmpreendimento')
    class_samples1 = class_samples1[class_samples1["CEG_GD"].isin(city_ventures['CodEmpreendimento'])].sort_values('CEG_GD')

    #print(class_samples1.values.shape, city_ventures.values.shape)

    columns = list(class_samples1.columns)+list(city_ventures.columns)
    resultdf:pd.DataFrame =pd.DataFrame(np.concatenate([class_samples1.values, city_ventures.values], 1), columns=columns)
    resultdf['NumCoordNEmpreendimento'] = resultdf['NumCoordNEmpreendimento'].str.replace(',', '.')
    resultdf['NumCoordEEmpreendimento'] = resultdf['NumCoordEEmpreendimento'].str.replace(',', '.')
    resultdf['MdaPotenciaInstaladaKW'] = resultdf['MdaPotenciaInstaladaKW'].str.replace(',', '.')
    resultdf['DthAtualizaCadastralEmpreend'] = resultdf['DthAtualizaCadastralEmpreend'].str.replace('-', '')

    i_sample:int = 4
    # geocode: 17
    #  coords: 18, 19
    #   power: 20
    #    date: -1

    print(resultdf.iloc[i_sample])
    
    dict_res = venture_gen(*resultdf.values[i_sample, [17, 18, 19, 20, -1]])

    date_str:str = '%02i/%02i/%04i'%(dict_res[2024][0, 0, 0]//1_0000%1_00, dict_res[2024][0, 0, 0]//1_00_0000%1_00, dict_res[2024][0, 0, 0]//1_0000_0000)
    
    month_period:np.ndarray = dict_res[2024][dict_res[2024][:, 0, 0]//1000000 == 202401, :, 1]
    week_day:int = datetime(2024, 1, 1).weekday()

    def week_day_load(d:np.ndarray) -> np.ndarray:
        r = d.copy()
        r[d==5] = 2
        r[d==6] = 0
        r[d<5] = 1

        return r
    
    sample_class_curves:pd.DataFrame = curve_per_class[curve_per_class['COD_ID'] == resultdf['TIP_CC'][i_sample]]
    
    print(np.concatenate(
        [sample_class_curves[sample_class_curves['TIP_DIA'] == 'DU'].iloc[:, 3:27].values.astype(np.float64)*20,
        sample_class_curves[sample_class_curves['TIP_DIA'] != 'DU'].iloc[:, 3:27].values.astype(np.float64)*4,
        sample_class_curves.iloc[week_day_load((np.arange(month_period.shape[0]%28)+week_day)%7), 3:27].values.astype(np.float64)],
    axis=0).shape)

    month_gross_load:np.ndarray = np.sum(np.concatenate(
        [sample_class_curves[sample_class_curves['TIP_DIA'] == 'DU'].iloc[:, 3:27].values.astype(np.float64)*20,
        sample_class_curves[sample_class_curves['TIP_DIA'] != 'DU'].iloc[:, 3:27].values.astype(np.float64)*4,
        sample_class_curves.iloc[week_day_load((np.arange(month_period.shape[0]%28)+week_day)%7), 3:27].values.astype(np.float64)],
    axis=0)/1000, 0)

    month_pv_gen:np.ndarray = np.sum(dict_res[2024][dict_res[2024][:, 0, 0]//1000000 == 202401, :, 1], 0)/1000

    """ plt.title('%s - %s\nDisaggregation Curves\n%s-%s\n\nInstalled PV Capacity: %s kW\nGross Load: %.2f kWh    Generation: %.2f kWh\n\nLiquid Load: %.2f kWh    Concessionaire Load Report: %s'%(
        resultdf.iloc[i_sample, 15], resultdf['TIP_CC'][i_sample],
        date_str[3:5], date_str[6:], 
        resultdf.iloc[i_sample, 20],
        np.sum(month_gross_load), np.sum(month_pv_gen),
        np.sum(month_gross_load-month_pv_gen), resultdf['ENE_01'][i_sample]
    ))
    plt.plot(np.arange(24), month_gross_load, label='Gross Load')
    plt.plot(np.arange(24), month_pv_gen, label='Generation')
    plt.plot(np.arange(24), month_gross_load-month_pv_gen, '--', label='Liquid Load')
    plt.legend()
    plt.tight_layout()
    plt.show()

    return """
    gross_consumption:np.ndarray = (curve_per_class[curve_per_class['COD_ID'] == resultdf['TIP_CC'][i_sample]].iloc[1, 3:27].values.astype(np.float64))/1000
    generation:np.ndarray = (dict_res[2024][0, :, 1])/1000
    net_consumption:np.ndarray = (gross_consumption-generation)
    self_consumtion:np.ndarray = generation.copy()
    idxs:np.ndarray = gross_consumption<generation
    self_consumtion[idxs] = gross_consumption[idxs]



    plt.title('%s\n\n%s  -  %s\n\n Installed PV Capacity: %s kW\n\nPV Generation: %.2f kWh   Gross Consumption: %.2f kWh   Net Consumption: %.2f kWh   Self-Consumption: %.2f kWh (%.2f%%)'%(
        resultdf['TIP_CC'][i_sample], resultdf.iloc[i_sample, 15], date_str, resultdf.iloc[i_sample, 20], np.sum(generation), np.sum(gross_consumption), np.sum(net_consumption), np.sum(self_consumtion), np.sum(self_consumtion)/np.sum(gross_consumption)*100))
    plt.plot(np.arange(24), generation*1000, label='PV Generation')
    plt.plot(np.arange(24), gross_consumption*1000, label='Gross Consumption')
    plt.plot(np.arange(24), net_consumption*1000, '--', label='Net Consumption')
    plt.xlabel('Hour')
    plt.ylabel('Power [W]')
    plt.legend(loc='upper left')
    plt.tight_layout()

    plt.show()


    """ #Ventures build
    t0:float = perf_counter()
    states_cities_coords_array:defaultdict[str, defaultdict[str, dict[str, np.ndarray]]] = ventures_process()
    print('Ventures process execution time:', perf_counter()-t0)
    state:defaultdict[str, defaultdict[str, dict[str, np.ndarray]]] = states_cities_coords_array['SP']
    monolith:bool = True

    coord__date_power_qty_array:dict[str, np.ndarray] = {}
    venture_gen(geocode, 'asas', 123354564, 156456.) """


def main() -> None:
    disaggregation_curve()

if '__main__' == __name__:
    main()