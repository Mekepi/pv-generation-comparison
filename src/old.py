import numpy as np
from time import perf_counter
from datetime import datetime
from os import listdir
from os.path import dirname, abspath, isfile
from pathlib import Path
from scipy.spatial import cKDTree
from psutil import cpu_count
from multiprocessing import Pool
from gzip import open as gzopen
from os import makedirs
from collections import defaultdict

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

def loss(g:float, t:float, install_year:int) -> float:
    Tc:float = (t+g*25/800)
    Tc = Tc if Tc>25 else 25

    return 1-0.95*0.97*0.98*(1-0.005*(datetime.now().year-install_year))*(1 - 0.0045*(Tc-25))

def average_day_radiation_plot(month:np.ndarray, venture_coord:list[float], data:list[str], city_plot_folder:Path, plot_type:str):
    ceg:str = data[0]
    geocode:str = data[1]
    power:float = float(float('.'.join(data[4].split(','))))*1000
    panels:int = int(data[6])
    area:float = float('.'.join(data[5].split(',')))
    install_year:int = int(''.join(data[27][:4]))


    hourly_radiation:np.ndarray = np.sum(month[:,:, 0], 0)/month.shape[0]

    hourly_generation:np.ndarray = np.zeros((24,2))
    l0:np.ndarray = np.vectorize(loss)(month[:, :, 0], month[:, :, 1], install_year)

    hourly_generation[:,0] = np.sum(np.vectorize(lambda g, l: g*power*(1-l)/1000)(month[:,:,0], l0), 0)/month.shape[0]
    hourly_generation[:,1] = np.sum(l0, 0)/month.shape[0]

    # v2:
    radiation_energy:float = np.sum(hourly_radiation, 0, float)/1000*area # [kWh]
    loss1:float = np.average(hourly_generation[hourly_generation[:,0]>0., 1])
    correction_factor1:float = power*(1-loss1)/1000
    generated_energy1:float = np.sum(hourly_generation[:, 0], 0, float)/1000 # [kWh]

    """ loss2:float = 0.14
    print([power*(1-loss2)/h for h in hourly_radiation if h>0])
    correction_factor2:float = sum([power*(1-loss2)/h for h in hourly_radiation if h>0])/hourly_radiation[hourly_radiation>0].shape[0]
    generated_energy2:float = sum(hourly_radiation)*correction_factor2/1000 """


    import matplotlib.pyplot as plt
    from matplotlib import use
    from matplotlib.axes import Axes
    use("Agg")

    ax1:Axes
    fig, ax1 = plt.subplots()

    ax2:Axes = ax1.twinx() #type: ignore
    
    ax2.plot(range(24), hourly_radiation, label="Irradiance Curve", color="black") #radiation
    ax1.bar(range(24), hourly_generation[:,0], label="Generation Bar", color="#24B351") # 1sr formula
    #ax1.bar(range(24), hourly_radiation*correction_factor2, label="Method 2", color="#1F2792", alpha=0.70) #2nd formula
    plt.title("Average Day PV Yield\n"+
              "%s\n\n"%(plot_type)+
              
              "%s\n"%(ceg)+
              "(%.6f,%.6f) [%s]\n\n"%(venture_coord[1], venture_coord[0], geocode)+

              "Panels: %i    Panels total area: %.0f m²    Total installed power: %.0f kW\n"%(panels, area, power/1000)
              )
    #plt.suptitle("PV Yield at (%f,%f), [%s]\n\nArea: %.0f m²    Panels: %i    Power: %.0f kW"%(venture_coord[1], venture_coord[0], geocode, area, panels, power/1000))
    ax1.set_xlabel("Time [Hour]\n\nRadiation Energy: %.2f kWh\n\nFactor: %.3f    Produced Energy: %.2f kWh    Loss: %.2f %%"%(radiation_energy, correction_factor1, generated_energy1, loss1*100))
    ax1.set_ylabel("Energy [Wh]", color='#24B351')
    ax2.set_ylabel("Irradiance [W/m²]", color='black')
    
    if ax1.set_ylim()[1] > ax2.set_ylim()[1]:
        ax2.set_ylim(ax1.set_ylim())
    else:
        ax1.set_ylim(ax2.set_ylim())
    
    ax1.legend(loc=2)
    ax2.legend(loc=1)

    fig.tight_layout()
    plt.savefig("%s\\%s-%s.png"%(city_plot_folder, ceg, plot_type.split('(')[0][:-1]), backend='Agg', dpi=200)

    plt.close()

def year_radiation_plot(Z:np.ndarray, venture_coord:list[float], geocode:str, ceg:str, city_plot_folder:Path, plot_type:str):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D #type: ignore
    from matplotlib import use
    use("Agg")

    x:np.ndarray = np.array(list(range(1, 367)))
    y:np.ndarray = np.array(list(range(24)))

    Y, X = np.meshgrid(y, x)

    ax:Axes3D = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.view_init(20, -50, 0)
    ax.set_xlabel('Day of the Year [Day]')
    ax.set_ylabel('Hour of the Day [Hour]')
    ax.set_zlabel('Irradiance [W/m²]')
    ax.set_title("Hourly Solar Radiation Across the Year\n%s\n\n%s\n(%.6f,%.6f) [%s]"%(plot_type, ceg, venture_coord[1], venture_coord[0], geocode))
    plt.tight_layout()
    plt.savefig("%s\\%s-3D-year-radiation-%s.png"%(city_plot_folder, ceg, plot_type), backend='Agg', dpi=200)
    plt.close()

def curves_gen(venture_data:list[str], venture_coord:list[float], timeseries_coord:np.ndarray) -> None:

    timeseries_folder:Path = Path('%s\\data\\timeseries'%(Path(dirname(abspath(__file__))).parent))

    ceg:str = venture_data[0]
    geocode:str = venture_data[1]
    state:str = states[geocode[:2]]

    if (state == "AC"):
        time_correction:int = 5
    elif (state == "AM"):
        time_correction = 4
    elif (geocode == "2605459"):
        time_correction = 2
    else:
        time_correction = 3

    timeseries_path:Path = Path("%s\\%s\\[%s]\\%s"%(timeseries_folder, state, geocode,
                                                                next(f for f in listdir("%s\\%s\\[%s]"%(timeseries_folder, state, geocode)) if f[19:].startswith("(%.6f,%.6f)"%(timeseries_coord[1], timeseries_coord[0])))))
    
    with gzopen(timeseries_path, 'rt', encoding='utf-8') as f:
        lines:list[str] = f.readlines()[9:-12]

    
    avarege_year:defaultdict[str, list[list[float]]] = defaultdict(list[list[float]])
    for line in lines[:-(366*24)]:
        spline:list[str] = line.split(',')
        avarege_year[spline[0][4:8]].append([sum([float(j) for j in spline[1:4]]), float(spline[5])])
    
    last_year:list[list[list[float]]] = []
    day:list[list[float]] = []
    for line in lines[-(366*24):]:
        spline = line.split(',')
        avarege_year[spline[0][4:8]].append([sum([float(j) for j in spline[1:4]]), float(spline[5])])
        day.append([sum([float(j) for j in spline[1:4]]), float(spline[5])])
        if (len(day) == 24):
            last_year.append(day)
            day = []

    Z:np.ndarray = np.asarray([[[sum([avarege_year[day][j][0] for j in range(i, len(avarege_year[day]), 24)])/(len(avarege_year[day])/24),
                                 sum([avarege_year[day][j][1] for j in range(i, len(avarege_year[day]), 24)])/(len(avarege_year[day])/24)] for i in range(24)] for day in sorted(avarege_year.keys())])
    Z = np.concatenate([Z[:, time_correction:], Z[:, :time_correction]], axis=1)

    Zl:np.ndarray = np.asarray(last_year)
    Zl = np.concatenate([Zl[:, time_correction:], Zl[:, :time_correction]], axis=1)

    
    city_plot_folder:Path = Path("%s\\outputs\\plot\\%s\\[%s]"%(Path(dirname(abspath(__file__))).parent, state, geocode))
    makedirs("%s"%(city_plot_folder), exist_ok=True)
    
    year_radiation_plot(Z[:,:,0], venture_coord, geocode, ceg, city_plot_folder, 'Average Year (%s;%s)'%(lines[0][:4], lines[-1][:4]))
    year_radiation_plot(Zl[:,:,0], venture_coord, geocode, ceg, city_plot_folder, 'Year of %s'%(lines[-1][:4]))
     
    max_average_month:np.ndarray = np.zeros([31,24,2])
    min_average_month:np.ndarray = np.ones([31,24,2])*np.inf
    max_av_i = -1
    min_av_i = -1


    max_last_year_month:np.ndarray = np.zeros([31,24,2])
    min_last_year_month:np.ndarray = np.ones([31,24,2])*np.inf
    max_last_i = -1
    min_last_i = -1

    days_of_mounths:list[int] = [31,29,31,30,31,30,31,31,30,31,30,31]
    i:int = 0
    for j in range(12):
        d:int = days_of_mounths[j]
        current_sum:float = np.sum(Z[i:i+d, :, 0])/d
        if (current_sum >= np.sum(max_average_month)/max_average_month.shape[0]):
            max_average_month = Z[i:i+d, :, :]
            max_av_i = j
        if (current_sum <= np.sum(min_average_month)/min_average_month.shape[0]):
            min_average_month = Z[i:i+d, :, :]
            min_av_i = j

        current_sum = np.sum(Zl[i:i+d, :, 0])/d
        #print("last: ", current_sum)
        if (current_sum >= np.sum(max_last_year_month)/max_last_year_month.shape[0]):
            max_last_year_month = Zl[i:i+d, :, :]
            max_last_i = j
        if (current_sum <= np.sum(min_last_year_month)/min_last_year_month.shape[0]):
            min_last_year_month = Zl[i:i+d, :, :]
            min_last_i = j

        i += d
    
    """ print('\n',
        "max average: %.2f %s"%(np.sum(max_average_month)/max_average_month.shape[0], max_average_month.shape), '\n',
        "max average: %.2f %s"%(np.sum(min_average_month)/min_average_month.shape[0], min_average_month.shape), '\n',
        "max average: %.2f %s"%(np.sum(max_last_year_month)/max_last_year_month.shape[0], max_last_year_month.shape), '\n',
        "max average: %.2f %s"%(np.sum(min_last_year_month)/min_last_year_month.shape[0], min_last_year_month.shape), '\n'
    ) """

    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December"
    ]

    for i in range(4):
        ploting = [max_average_month, min_average_month, max_last_year_month, min_last_year_month][i]
        ploting_i = [max_av_i, min_av_i, max_last_i, min_last_i][i]
        header:str = ['Average Year Max Month (%s)'%(months[ploting_i]),
                      'Average Year Min Month (%s)'%(months[ploting_i]),
                      'Last Year Max Month (%s, %s)'%(months[ploting_i], lines[-1][:4]),
                      'Last Year Min Month (%s, %s)'%(months[ploting_i], lines[-1][:4])][i]
        
        average_day_radiation_plot(ploting, venture_coord, venture_data, city_plot_folder, header)