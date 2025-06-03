import numpy as np
from os.path import dirname, abspath, isfile
from pathlib import Path
from psutil import cpu_count
from multiprocessing import Pool
import pickle

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

def state_usines_pv_mmd_generation(state:str) -> tuple[str, np.ndarray]:
    with open('%s\\data\\usine_generation\\GERACAO_USINA-2_2000-2024_%s.csv'%(Path(dirname(abspath(__file__))).parent, state), 'r', 512*1024, encoding='utf-8') as f:
        lines:list[str] = f.readlines()[1:]
    
    col0:list[str] = []
    col1:list[str] = []
    for line in lines:
        spline:list[str] = line.split(';', 1)
        col0.append(spline[0][:10].replace('-', '', 2))
        col1.append(spline[1].replace('.', ''))
    
    Z:np.ndarray = (np.array([col0, col1], np.float64).T).reshape((len(col0)//24, 24, 2))
    
    return (state, Z)

def usine_plot(state:str, Z0:np.ndarray, period:tuple[int,int] = (0,0)) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D #type: ignore
    from matplotlib.ticker import FixedLocator, FixedFormatter

    if period == (0,0):
        period = (int(Z0[0, 0, 0]), int(Z0[-1, 0, 0]))

    i0:int = int(np.argwhere(Z0[:,0,0] >= period[0])[0, 0])
    i:int = int(np.argwhere(Z0[:,0,0] <= period[1])[-1, 0])
    Z:np.ndarray = Z0[i0:i+1]
    Z[:, :, 1] = Z[:, :, 1]/(10**6)

    period = (int(Z[0, 0, 0]), int(Z[-1, 0, 0]))

    print('usines:', period, Z[[0,-1], 0,  0].astype(np.str_))

    x:np.ndarray = np.arange(1, Z.shape[0]+1)
    y:np.ndarray = np.arange(Z.shape[1])

    Y, X = np.meshgrid(y, x)

    Z_max_Y:np.ndarray = np.max(Z[:, :, 1], axis=1)
    from scipy.ndimage import gaussian_filter1d
    smoothed_Z_max_Y = gaussian_filter1d(Z_max_Y, sigma=10)

    ax:Axes3D = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z[:,:,1], cmap='viridis')
    ax.plot(x, [y.max()]*len(x), smoothed_Z_max_Y, color='#440154', linewidth=2, label='Smoothed Z Max over X')
    ax.view_init(20, -50, 0)
    
    all_idxs:list[int] = list(np.unique(Z[:, 0, 0]//1_00, True)[1].astype(int))
    idxs:list[int] = [all_idxs[i] for i in range(0, len(all_idxs), len(all_idxs)//5+1)]
    month_labels = ['%04i-%02i'%(Z[i, 0, 0]//1_0000, Z[i, 0, 0]%1_0000/1_00) for i in idxs]

    # Set the locations of the ticks to correspond to the first day of each unique month
    x_locator = FixedLocator(idxs)
    ax.xaxis.set_major_locator(x_locator)

    # Set the labels for these ticks using the month strings
    x_formatter = FixedFormatter(month_labels)
    ax.xaxis.set_major_formatter(x_formatter)

    ax.set_xlabel('Year-Month [YYYY-MM]')
    ax.set_ylabel('Hour of the Day [Hour]')
    ax.set_zlabel('Power [MW]')
    ax.set_title("Usines Generation Across (%02i/%i:%02i/%i)\n[%s]\n\nTotal produced: %.2f TWh"%(period[0]%10000//100, period[0]//10000, period[1]%10000//100, period[1]//10000, state, np.sum(Z[:,:,1])/(10**6)))
    plt.tight_layout()
    #plt.show()
    plt.savefig("%s\\outputs\\Usines MMD PV Generation\\usines-%s-(%i_%02i, %i_%02i).png"%(Path(dirname(abspath(__file__))).parent, state, period[0]//10000, period[0]%10000//100, period[1]//10000, period[1]%10000//100), backend='Agg', dpi=200)
    plt.close()

def usines_plot(state_Z_list:list[tuple[str, np.ndarray]]) -> None:
    with Pool(cpu_count()) as p:
        p.starmap(usine_plot, state_Z_list)

def usines_pv_mmd_generation() -> dict[str, np.ndarray]:
    if (isfile('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\per_state_usines_pv_mmd_generation.pkl'))):
        with open('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\per_state_usines_pv_mmd_generation.pkl'), 'rb', 8*1024*1024) as fin:
            return pickle.load(fin)

    with Pool(cpu_count()) as p:
        result:list[tuple] = p.map(state_usines_pv_mmd_generation, list(states.values()))

    per_state_usines_pv_mmd_generation:dict[str, np.ndarray] = {state:array for state, array in result}

    with open('%s\\%s'%(Path(dirname(abspath(__file__))).parent, 'data\\pickles\\per_state_usines_pv_mmd_generation.pkl'), 'wb', 8*1024*1024) as fout:
        pickle.dump(per_state_usines_pv_mmd_generation, fout)

    return per_state_usines_pv_mmd_generation