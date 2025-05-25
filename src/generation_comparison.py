import numpy as np
from time import perf_counter
from os.path import dirname, abspath, isfile
from pathlib import Path
from psutil import cpu_count
from multiprocessing import Pool
from gzip import open as gzopen
from os import makedirs
from collections import defaultdict
import pickle

from usine_generation import usine_plot, usines_plot, usines_pv_mmd_generation
from venture_generation import ventures_process, save_ventures_timeseries_coords_filtered

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

#get and plot ventures generation

def coord_process(geocode:str, coord:str, date_power_qty_array:np.ndarray) -> dict[int, np.ndarray]:
    timeseries_path:Path =  Path(dirname(abspath(__file__))).parent.joinpath('data\\timeseries\\%s\\[%s]\\[%s]timeseries%s.csv.gz'%(states[geocode[:2]], geocode, geocode, coord))
    with gzopen(timeseries_path, 'rt', encoding='utf-8') as f:
        lines:list[str] = f.readlines()[9:-12]

    min_year:int = min([int(date_power_qty_array[0,0]), 2020])
    timeseries_array:np.ndarray = np.zeros((sum([366 if i%4==0 else 365 for i in range(min_year, 2024)])*24, 3), np.float64)
    i0:int = sum([366 if i%4==0 else 365 for i in range(2005, min_year)])*24
    j:int = 0
    for line in lines[i0:]:
        spline:list[str] = line.split(',', 6)[:-1]
        timeseries_array[j, 0] = spline[0].replace(':', '')
        timeseries_array[j, 1] = sum([float(v) for v in spline[1:4]])
        timeseries_array[j, 2] = spline[5]
        j += 1

    
    arr2024:np.ndarray = np.zeros((366*24, 3))

    arr2024[:, 1:] += timeseries_array[timeseries_array[:, 0]//(10**8) == 2020][:, 1:]
    for i in [2021, 2022, 2023]:
        arr2024[:59*24, 1:] += timeseries_array[timeseries_array[:, 0]//(10**8) == i][:59*24, 1:]
        arr2024[60*24:, 1:] += timeseries_array[timeseries_array[:, 0]//(10**8) == i][59*24:, 1:]
        arr2024[59*24:60*24, 1:] += (timeseries_array[timeseries_array[:, 0]//(10**8) == i][58*24:59*24, 1:]+timeseries_array[timeseries_array[:, 0]//(10**8) == i][59*24:60*24, 1:])/2
    arr2024[:, 1:] /= 4

    arr2024[:, 0] = timeseries_array[timeseries_array[:, 0]//(10**8) == 2020][:, 0]%(10**8)
    arr2024[:, 0] += 2024*(10**8)
    
    timeseries_array = np.concatenate((timeseries_array, arr2024))


    def t_correction(g:np.ndarray, t2m:np.ndarray) -> np.ndarray:
        #Tc:np.ndarray = (t2m+g*25/800)
        #Tc[Tc<25] = 25
        
        #return (1-0.0045*(Tc-25))
        return (1-0.0045*(np.maximum((t2m+g*25/800), 25)-25))
    
    Z0:dict[int, np.ndarray] = {}
    current_power:float = 0.
    for year in range(min_year, int(timeseries_array[-1, 0]//(10**8))+1):
        current_power = current_power*0.995 + (date_power_qty_array[date_power_qty_array[:, 0]==year][0, 1]*1000*0.95*0.97*0.98 if (date_power_qty_array[date_power_qty_array[:, 0]==year].size>0) else 0)
        
        year_mask:np.ndarray = timeseries_array[:, 0]//(10**8) == year
        
        Z0[year] = (timeseries_array[year_mask, 1]*current_power*t_correction(timeseries_array[year_mask, 1], timeseries_array[year_mask, 2])/1000).reshape((366 if year%4==0 else 365, 24))
    
    return Z0 

def coord_process_monolith(geocode:str, coord:str, date_power_qty_array:np.ndarray) -> dict[int, np.ndarray]:
    if (int(date_power_qty_array[0,0])>20241231):
        return {}
    
    timeseries_path:Path =  Path('C:\\Programas\\timeseries\\%s\\monolith\\[%s]timeseries%s.csv.gz'%(states[geocode[:2]], geocode, coord))
    with gzopen(timeseries_path, 'rt', encoding='utf-8') as f:
        lines:list[str] = f.readlines()[9:-12]

    min_year:int = min([int(date_power_qty_array[0,0])//10000, 2020])
    i0:int = sum([366 if i%4==0 else 365 for i in range(2005, min_year)])*24

    col0:list[str] = []
    col1:list[float] = []
    col2:list[str] = []
    for line in lines[i0:]:
        spline = line.split(',', 6)
        col0.append(spline[0].replace(':', '', 1))
        col1.append(float(spline[1])+float(spline[2])+float(spline[3]))
        col2.append(spline[5])
    timeseries_array:np.ndarray = np.array([col0, col1, col2], np.float64).T

    
    year_col:np.ndarray = timeseries_array[:, 0 ].astype(np.int64)
    val_cols:np.ndarray = timeseries_array[:, 1:].astype(np.float64)

    year_fil:np.ndarray = year_col//(1_0000_0000)
    mask:list[np.ndarray] = [
        year_fil == 2020,
        year_fil == 2021,
        year_fil == 2022,
        year_fil == 2023
    ]

    c59_24:int = 59*24
    c60_24:int = 60*24

    arr2024:np.ndarray = np.zeros((366*24, 3), np.float64)

    arr2024[:, 1:] += val_cols[mask[0]]
    for i in [1, 2, 3]:
        year_vals:np.ndarray = val_cols[mask[i]]
        arr2024[:c59_24, 1:] += year_vals[:c59_24]
        arr2024[c60_24:, 1:] += year_vals[c59_24:]
        arr2024[c59_24:c60_24, 1:] += np.mean([
            year_vals[c59_24-24:c59_24],
            year_vals[c59_24:c60_24]
        ], 0)
    
    arr2024 /= 4

    arr2024[:, 0] = 2024*(1_0000_0000) + year_col[mask[0]]%(1_0000_0000)

    timeseries_array = np.concatenate([timeseries_array, arr2024], 0)

    
    if (states[geocode[:2]] == 'AC'):
        time_correction:int = 5
    elif (states[geocode[:2]] == 'AM'):
        time_correction = 4
    elif (geocode == '2605459'):
        time_correction = 2
    else:
        time_correction = 3
    
    timeseries_mask_arg:int = np.argwhere(timeseries_array[:, 0]//(10**4) == date_power_qty_array[0, 0])[0, 0]
    
    timeseries_array[timeseries_mask_arg:, 1:] = np.concatenate([
        timeseries_array[timeseries_mask_arg+time_correction:, 1:],
        timeseries_array[timeseries_mask_arg:timeseries_mask_arg+time_correction, 1:]
    ], 0)

    date_irradiance_t2m_array = timeseries_array[timeseries_mask_arg:].reshape(timeseries_array[timeseries_mask_arg:].shape[0]//24, 24, 3)


    def t_correction(g:np.ndarray, t2m:np.ndarray) -> np.ndarray:
        #Tc:np.ndarray = (t2m+g*25/800)
        #Tc[Tc<25] = 25
        
        #return (1-0.0045*(Tc-25))
        return (1-0.0045*(np.maximum((t2m+g*25/800), 25)-25))

    date_power_qty_array[:, 1] = (np.round(date_power_qty_array[:, 1].astype(np.float64)*0.95*0.97*0.98, 0)).astype(np.int32)

    power_array:np.ndarray = np.zeros((date_irradiance_t2m_array.shape[0], 24, 1), np.float64)

    for i in range(date_power_qty_array[date_power_qty_array[:, 0] < 20250101][:, 0].shape[0]):
        j = np.argwhere(date_irradiance_t2m_array[:, 0, 0]//(10**4) == date_power_qty_array[i, 0])[0, 0]
        power_array[j:, :] +=(date_power_qty_array[i, 1]*np.power(0.995, np.arange(power_array[j:].shape[0])/365.25))[:, np.newaxis, np.newaxis]
        

    energy_array:np.ndarray = np.concatenate([
        date_irradiance_t2m_array[:, :, :1],
        (date_irradiance_t2m_array[:, :, 1:2]*power_array*t_correction(date_irradiance_t2m_array[:, :, 1:2], date_irradiance_t2m_array[:, :, 2:]))/1000
    ], 2)
    
    Z0:dict[int, np.ndarray] = {year:energy_array[energy_array[:, 0, 0]//(10**8) == year] for year in range(int(date_irradiance_t2m_array[0, 0, 0]//(10**8)), int(date_irradiance_t2m_array[-1, 0, 0]//(10**8))+1)}
    
    return Z0 

def plot_generation(state:dict[str, dict[str, np.ndarray]], period:tuple[int, int] = (0, 0), monolith:bool = False) -> None:
    state_abbreviation:str = states[list(state.keys())[0][:2]]

    preZ:dict[int, np.ndarray]
    if (not(isfile('%s\\data\\pickles\\preZ.pkl'%(Path(dirname(abspath(__file__))).parent)))):
        preZ = {}
        with Pool(cpu_count()) as p:
            if monolith:
                all_coords_dict:dict[str, tuple[str, np.ndarray]] = {key:(geocode, coords_dict[key][coords_dict[key][:,0].argsort()]) for geocode, coords_dict in state.items() for key in coords_dict.keys()}
                
                Z0s:list[dict[int, np.ndarray]] = p.starmap(
                    coord_process_monolith,
                    [(geocode__date_power_qty_array[0], coord, geocode__date_power_qty_array[1]) for coord, geocode__date_power_qty_array in all_coords_dict.items()]
                )

                for Z0 in Z0s:
                    for year, array in Z0.items():
                        if year not in preZ:
                            preZ[year] = np.zeros((366 if year%4==0 else 365, 24, 2), np.float64)
                        preZ[year][-array.shape[0]:, :, 1] += array[:, :, 1]
                        preZ[year][-array.shape[0]:, :, 0]  = array[:, :, 0]

            else:
                    for geocode, coords_dict in state.items():
                        Z0s = p.starmap(coord_process, [(geocode, coord, date_power_qty_array) for coord, date_power_qty_array in coords_dict.items()])
                        
                        for Z0 in Z0s:
                            for year, array in Z0.items():
                                if year not in preZ:
                                    preZ[year] = np.zeros((366 if year%4==0 else 365, 24, 2))
                                preZ[year][-array.shape[0]:] += array

            with open('%s\\data\\pickles\\preZ.pkl'%(Path(dirname(abspath(__file__))).parent), 'wb', 4*(1024**2)) as fout:
                pickle.dump(preZ, fout)
    else:
        with open('%s\\data\\pickles\\preZ.pkl'%(Path(dirname(abspath(__file__))).parent), 'rb', 4*(1024**2)) as fin:
            preZ = pickle.load(fin)

    #all_coords_dict = {coord:coords_dict[coord][coords_dict[coord][:,0].argsort()] for coords_dict in state.values() for coord in coords_dict.keys()}
    #print(*['%4i, %5.2fMW, %5.2fMWh'%(key, sum([np.sum(array[array[:, 0]<=key][:, 1]) for array in all_coords_dict.values()])/(10**3), np.sum(preZ[key])/(10**6)) for key in sorted(preZ.keys())], sep='\n')

    #print([(key, np.round(np.sum(preZ[key])/(10**6),2)) for key in sorted(list(preZ.keys()))]) #(year, installed power [MW])

    if period == (0,0):
        period_list:list[int] = sorted(list(preZ.keys()))
        period = (period_list[0]*(10**4)+101, period_list[-1]*(10**4)+1231)

    Zm:np.ndarray = np.concatenate([preZ[key] for key in sorted(list(preZ.keys()))])

    i0:int = np.argwhere(Zm[:,0,0]//(10**4) >= period[0])[0, 0]
    i:int = np.argwhere(Zm[:,0,0]//(10**4) <= period[1])[-1, 0]
    Z:np.ndarray = Zm[i0:i+1]
    Z[:, :, 1] = Z[:, :, 1]/(10**6)

    print('ventures:', period, Z[[0,-1], 0,  0])

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D #type: ignore

    x:np.ndarray = np.arange(1, Z.shape[0]+1)
    y:np.ndarray = np.arange(Z.shape[1])

    Y, X = np.meshgrid(y, x)

    Z_max_Y:np.ndarray = np.max(Z[:, :, 1], axis=1)
    from scipy.ndimage import gaussian_filter1d
    smoothed_Z_max_Y = gaussian_filter1d(Z_max_Y, sigma=10)

    ax:Axes3D = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z[:, :, 1], cmap='viridis')
    ax.plot(x, [y.max()]*len(x), smoothed_Z_max_Y, color='#440154', linewidth=2, label='Smoothed Z Max over X')
    ax.view_init(20, -50, 0)
    ax.set_xlabel('Day of the Data [Day]')
    ax.set_ylabel('Hour of the Day [Hour]')
    ax.set_zlabel('Power [MW]')
    ax.set_title('Ventures PV Yield Across (%02i/%i:%02i/%i)\n[%s]\n\nTotal produced: %.2fTWh'%(period[0]%10000//100, period[0]//10000, period[1]%10000//100, period[1]//10000, state_abbreviation, np.sum(Z[:, :, 1])/10**6))
    plt.tight_layout()
    #plt.show()
    plt.savefig("%s\\outputs\\Ventures MMD PV Generation\\ventures-%s-(%i_%02i, %i_%02i).png"%(Path(dirname(abspath(__file__))).parent, state_abbreviation, period[0]//10000, period[0]%10000//100,  period[1]//10000, period[1]%10000//100), backend='Agg', dpi=200)
    plt.close()

def main() -> None:

    #Ventures build
    t0:float = perf_counter()
    states_cities_coords_array:defaultdict[str, defaultdict[str, dict[str, np.ndarray]]] = ventures_process()
    print('Ventures process execution time:', perf_counter()-t0)

    #states_cities_coords_array comprehension
    #print(states_cities_coords_array.keys())
    #print(states_cities_coords_array['SP'].keys())
    #print(sorted([(coord, array.shape[0])for coord, array in states_cities_coords_array['SP']['3500105'].items()], key=lambda e: e[1]))
    #print(states_cities_coords_array['SP']['3500105']['(-21.693625,-51.070219)'])
    #print([date for date in states_cities_coords_array['SP']['3500105']['(-21.693625,-51.070219)'][:, 0] if not(2009<=date//10000<=2025 and 1<=(date%10000)//100<=12 and 1<=date%100<=31)])
    
    #Save filtered coords
    """ t0 = perf_counter()
    save_ventures_timeseries_coords_filtered(states_cities_coords_array, True)
    print('Save filtered timeseries coords time:', perf_counter()-t0) """

    #Ventures timeseries data volume analysis
    """ total:int = 0
    for state, cities in states_cities_coords_array.items():
        subtotal:int = sum([len(city.keys()) for city in cities.values()])
        total += subtotal
        print('%s: %5i timeseries coords (%5.2f GiB compressed -> %5.1fmin)'%(state, subtotal, 1650*subtotal/(1024**2), 0.42981*subtotal/60))
    print('Total space required: %i timeseries (%.2f GiB after compression)'%(total, 1650*total/(1024**2)))
    print('Total minimum expected download time: Uncompressed %.2f TiB -> %.1fh'%(7.65*total/(1024**2), 0.42981*total/(60**2))) """

    #Usines build
    t0 = perf_counter()
    per_state_usines_pv_mmd_generation:dict[str, np.ndarray] = usines_pv_mmd_generation()
    print('\nUsines process execution time:', perf_counter()-t0)

    ##################################################################################################

    period:tuple[int, int] = (20230429, 20241231)
    #period:tuple[int, int] = (20240000+m*100, 20240000+m*100+99)

    #Ventures plot // depends on having timeseries
    t0 = perf_counter()
    plot_generation(states_cities_coords_array['SP'], period, monolith=True)
    print('Ventures generetaion plot execution time:', perf_counter()-t0)

    #Usines plot
    t0 = perf_counter()
    usine_plot('SP', per_state_usines_pv_mmd_generation['SP'], period)
    print('Usines generation plot execution time:', perf_counter()-t0)

if __name__ == "__main__":
    main()

# (0, '"CodEmpreendimento') (1, 'CodMunicipioIbge') (2, 'NumCoordNEmpreendimento') (3, 'NumCoordEEmpreendimento')
# (4, 'MdaPotenciaInstaladaKW') (5, 'MdaAreaArranjo') (6, 'QtdModulos') (7, 'MdaPotenciaModulos') (8, 'NomModeloModulo')
# (9, 'NomFabricanteModulo') (10, 'MdaPotenciaInversores') (11, 'NomModeloInversor"') (12, 'NomFabricanteInversor')
# (13, 'QtdUCRecebeCredito') (14, 'CodClasseConsumo') (15, 'DscSubGrupoTarifario') (16, 'SigModalidadeEmpreendimento')
# (17, 'SigTipoConsumidor') (18, 'NumCPFCNPJ') (19, 'CodCEP') (20, 'NumCNPJDistribuidora') (21, 'NomAgente')
# (22, 'NomSubEstacao') (23, 'NumCoordNSub') (24, 'NumCoordESub') (25, 'SigTipoGeracao') (26, 'DscPorte') (27, 'DthAtualizaCadastralEmpreend"\n')