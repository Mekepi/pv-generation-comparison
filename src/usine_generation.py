import numpy as np
from os.path import dirname, abspath
from pathlib import Path
from psutil import cpu_count
from multiprocessing import Pool

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}


def usine_plot(state:str, Z:np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D #type: ignore
    from matplotlib import use
    use("Agg")

    x:np.ndarray = np.arange(1, Z.shape[0]+1)
    y:np.ndarray = np.arange(Z.shape[1])

    Y, X = np.meshgrid(y, x)

    ax:Axes3D = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z[:,:,1], cmap='viridis')
    ax.view_init(20, -50, 0)
    ax.set_xlabel('Day of the Data [Day]')
    ax.set_ylabel('Hour of the Day [Hour]')
    ax.set_zlabel('Power [MW]')
    ax.set_title("Usines Generation Across (%02i/%i:%02i/%i)\n[%s]\n\nTotal produced: %.2f TWh"%(Z[0,0,0]%10000//100, Z[0,0,0]//10000, Z[-1,0,0]%10000//100, Z[-1,0,0]//10000, state, np.sum(Z[:,:,1])/(10**6)))
    plt.tight_layout()
    #plt.show()
    plt.savefig("%s\\outputs\\Usines MMD PV Generation\\usines-%s.png"%(Path(dirname(abspath(__file__))).parent, state), backend='Agg', dpi=200)
    plt.close()

def state_usines_pv_mmd_generation(state:str) -> tuple[str, np.ndarray]:
    with open('%s\\data\\usine_generation\\GERACAO_USINA-2_2000-2024_%s.csv'%(Path(dirname(abspath(__file__))).parent, state), 'r', 512*1024, encoding='utf-8') as f:
            lines:list[str] = f.readlines()[1:]
    
    Z:np.ndarray = np.asarray([[[float(''.join(lines[j].split(';')[0][:10].split('-'))), float(lines[j].split(';')[1])] for j in range(i, i+24)] for i in range(0, len(lines), 24)])

    return (state, Z)

def usines_pv_mmd_generation(sts:list[str] = []) -> dict[str, np.ndarray]:

    if not(sts):
        sts = list(states.values())

    with Pool(cpu_count()) as p:
        result:list[tuple] = p.map(state_usines_pv_mmd_generation, sts)

    return {state:array for state, array in result}