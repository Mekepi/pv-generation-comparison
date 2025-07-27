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

from venture_generation import ventures_process

states:dict[str, str] = {
    "12": "AC", "27": "AL", "13": "AM", "16": "AP", "29": "BA", "23": "CE", "53": "DF",
    "32": "ES", "52": "GO", "21": "MA", "31": "MG", "50": "MS", "51": "MT", "15": "PA",
    "25": "PB", "26": "PE", "22": "PI", "41": "PR", "33": "RJ", "24": "RN", "43": "RS",
    "11": "RO", "14": "RR", "42": "SC", "35": "SP", "28": "SE", "17": "TO"
}

def installed_capacity(t:np.ndarray) -> np.ndarray:
    base_growth = 22.36 + 0.1904 * (t / 365.25)
    
    return base_growth

def modified_installed_capacity(t:np.ndarray) -> np.ndarray:
    base_growth:np.ndarray = installed_capacity(t)
    
    return base_growth*(0.7+0.3/(1+np.exp(0.002*(t-7000))))

def pv_mmdg_installed_capacity(t:np.ndarray) -> np.ndarray:
    """PV installed capacity with smooth logistic transition at t==6350 (wich is the present) [GW]"""
    data:np.ndarray = t[t<=6350]
    logistic:np.ndarray = t[t>6350]

    x_n:np.ndarray = (1-np.power(2, -0.005*(logistic-5984.63)))

    print()

    return np.concatenate([5*(10**(-37))*np.power(data, 9.838), 0.7*installed_capacity(logistic)*(1-np.power(2, -0.005*(logistic-5984.63)))], 0)

def gen_dates(c_date:int, days:int) -> list[int]:
    ml = lambda y: [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] if y%4 == 0 else [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    dates:list[int] = [0]*days

    dates[0] =  c_date

    y:int
    m:int
    d:int
    y, m, d = (c_date//10000, (c_date//100)%100, c_date%100)
    
    i = ml(y)[m]-d+1 if ml(y)[m]-d+1<days else days
    
    dates[1:i] = list(range(c_date+1, c_date+i))[:i-1]
    
    def next_day(date:int) -> int:
        y,m,d = (date//10000, (date//100)%100, date%100)
        
        if (d < ml(y)[m]):
            return y*1_00_00+m*1_00+(d+1)
        
        elif (m < 12):
            return y*1_00_00+(m+1)*1_00+1
        
        else:
            return (y+1)*1_00_00+1*1_00+1
            
    days -= i
    c_date = next_day(dates[i-1])
    y,m,d = (c_date//10000, (c_date//100)%100, c_date%100)
    
    while days>=ml(y)[m]:
        dates[i:i+ml(y)[m]] = list(range(c_date, c_date+ml(y)[m]))
        i += ml(y)[m]
        days -= ml(y)[m]
        c_date = next_day(dates[i-1])
        y,m,d = (c_date//10000, (c_date//100)%100, c_date%100)
    
    dates[i:] = list(range(c_date, c_date+days))
    
    return dates

def installed_power_analysis(st='') -> None:
    
    if st and st in states.values():
        states_dicts:dict[str, dict] = {st:ventures_process()[st]}
    if st == 'BR':
        states_dicts = {'Brasil':{}}
        for sd in ventures_process().values():
            for ck, cd in sd.items():
                states_dicts['Brasil'][ck] = cd
    else:
        states_dicts = ventures_process()
    
    for st_key, st_dict in states_dicts.items():
        date_power:dict[np.int64, np.int64] = {}

        i:int
        for city in st_dict.values():
            for coord in city.values():
                for i in range(coord.shape[0]):
                    date:np.int64
                    power:np.int64
                    date, power, *_ = coord[i]
                    if date not in date_power:
                        date_power[date] = np.int64(0)
                    date_power[date] += power
        
        date_power_array:np.ndarray = np.sort(np.array(list(date_power.items()), np.int64), 0)

        ml:list[int]  = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        dates:np.ndarray = np.zeros((sum([366 if i%4==0 else 365 for i in range(date_power_array[0, 0]//1_0000, date_power_array[-1, 0]//1_0000+1)]), 1))

        i = 0
        for y in range((date_power_array[0, 0]//10000)*10000, (date_power_array[-1, 0]//10000)*10000+10000, 10000):
            for m in range(100, 1300, 100):
                for d in range(1, ml[m//100]+1):
                    dates[i, 0] = y+m+d
                    i = i+1
                    if m==200 and y%40000==0 and d==28:
                        dates[i, 0] = y+m+29
                        i = i+1
        
        i0:int = np.argwhere(dates==date_power_array[0, 0])[0, 0]
        i = np.argwhere(dates==date_power_array[-1, 0])[0, 0]
        dates = dates[i0:i+1]
        
        
        power_array:np.ndarray = np.zeros(dates.shape)
        lossless_power_array:np.ndarray = power_array.copy()

        js:np.ndarray = np.argwhere(dates[:, 0].astype(np.int64) == date_power_array[:, 0:1])[:, 1]
        
        for i in range(date_power_array.shape[0]):
            power_array[js[i]:, 0] += (date_power_array[i, 1]*np.power(0.995, np.arange(power_array[js[i]:].shape[0])/365.25))
            lossless_power_array[js[i]:, 0] += date_power_array[i, 1]
        
        ###########################################
        ###########################################

        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes
        from mpl_toolkits.mplot3d import Axes3D #type: ignore
        from matplotlib.ticker import FixedLocator, FixedFormatter

        Z:np.ndarray = np.concatenate((dates, power_array), 1)
        
        month_growth:list[float] = []
        month_label:list[float] = []
        m_number:float = (Z[0, 0]//100)%100
        m_d1_power:float = Z[0, 1]
        for d in range(Z.shape[0]):
            if (m_number == (Z[d, 0]//100)%100):
                continue
            month_growth.append(Z[d-1, 1]/m_d1_power)
            month_label.append(Z[d-1, 0]//100)
            m_number = (Z[d, 0]//100)%100
            m_d1_power = Z[d, 1]

        month_growth_array:np.ndarray = np.array(month_growth, np.float64)

        """ all_idxs:list[int] = list(np.unique(Z[:, 0]//1_00, True)[1])
        idxs:list[int] = [all_idxs[i] for i in range(0, len(all_idxs), 2)]
        month_labels = ['%04i'%(Z[:, 0][i]//1_00) for i in idxs] """

        idxs:list[int] = list(i for i in range(-12, 0, 2))
        month_labels = ['%02i-%04i'%(month_label[i]%100, month_label[i]//100) for i in idxs]

        axs:list[Axes]
        fig, axs = plt.subplots(2, 1)
        # Remove vertical space between Axes
        fig.subplots_adjust(hspace=5)

        # Plot each graph, and manually set the y tick values
        i0 = np.where(Z[:, 0]//100==month_label[-12])[0][0]
        axs[0].plot(np.arange(Z.shape[0]-i0), Z[i0:, 1]/10**9, )
        axs[0].xaxis.set_visible(False)
        axs[0].set_ylabel('Power [GW]')
        axs[0].set_title('PV MMDG Capacity Overview\n%s (%02i-%04i; %02i-%04i)\n\n Installed Power\n\n Current: %.2f GW'%(st_key, month_label[-12]%100, month_label[-12]//100, month_label[-1]%100, month_label[-1]//100, Z[-1, 1]/10**9))

        axs[1].plot(np.arange(12), (month_growth_array[-12:]-1)*100)
        axs[1].set_ylabel('Growth [%]')
        axs[1].set_title('Monthly Growth\n\n Mean: %.2f %%'%((np.mean(month_growth_array[-12:])-1)*100))


        # Set the locations of the ticks to correspond to the first day of each unique month
        x_locator = FixedLocator(np.arange(0, 12, 2))
        axs[1].xaxis.set_major_locator(x_locator)
        
        # Set the labels for these ticks using the month strings
        x_formatter = FixedFormatter(month_labels)
        axs[1].xaxis.set_major_formatter(x_formatter)

        plt.tight_layout()

        plt.savefig('%s\\outputs\\Installed Capacity Analysis\\%s-installed-capacity-analysis.png'%(Path(dirname(abspath(__file__))).parent, st_key), backend='agg', dpi=200)
        plt.close()

    return
    usedZ:np.ndarray = Z[:, 1]/(10**9)#*1.125
    current_mmdg_pv_capacity:float = float(usedZ[-1])

    x:np.ndarray = np.arange(1, Z.shape[0]+1)
    ax:Axes = plt.axes()
    """ with open('aqui.txt', 'w', 1000, 'utf-8') as fout:
        fout.writelines(['%f,%f\n'%(i, usedZ[i-1]) for i in x]) """
    
    usedZ = usedZ
    x = x

    x_ext:np.ndarray = np.linspace(1, 8000, 4000)
    Z_ext_date:np.ndarray = np.array(gen_dates(int(Z[0, 0]), int(x_ext[-1])))

    period:tuple[str, str] = ('%i'%(Z_ext_date[0]//1_00), '%i'%(Z_ext_date[-1]//1_00))

    current_date:str = '%02i/%04i'%(Z_ext_date[x.shape[0]]%10000//100, Z_ext_date[x.shape[0]]//10000)

    print('ventures:', period, Z[[0,-1],  0].astype(np.str_))

    print(x, usedZ, sep='\n')

    plt.plot(x_ext, installed_capacity(x_ext), label='Total Capacity', color='green')

    plt.plot(x_ext[x_ext>=6350], 0.7*installed_capacity(x_ext[x_ext>=6350]), label='0.7*Total Capacity', color='black', alpha=0.2)
    
    trendZ:np.ndarray = np.polyfit(np.log10(x[dates[:, 0]>=20200000]), np.log10(usedZ[dates[:, 0]>=20200000]), 1)
    print(trendZ)
    #y = %.0e · $t^{%.3f}$'%(10**trendZ[1], trendZ[0])
    plt.scatter(x, usedZ, label='MMDG PV Capacity')
    plt.plot(x_ext, (10**trendZ[1])*np.power(x_ext, trendZ[0]), label='MMDG PV trend', color='orange')

    plt.plot(x_ext, pv_mmdg_installed_capacity(x_ext), label='MMDG PV logistic', color='red')

    current_x:float = x[-1]

    plt.plot(current_x, installed_capacity(np.array([current_x]))[0], marker='o', linestyle='-', color='black')
    plt.annotate('Current Total Capacity', (current_x, installed_capacity(np.array([current_x]))[0]), xytext=(-5, 5), textcoords='offset points', ha='right')
    plt.annotate('%.2f GW'%(installed_capacity(np.array([current_x]))[0]), (current_x, installed_capacity(np.array([current_x]))[0]), xytext=(0, -15), textcoords='offset points', ha='left')
    
    plt.plot(current_x, current_mmdg_pv_capacity, marker='o', linestyle='-', color='black')
    plt.annotate('Current MMDG PV Capacity', (current_x, current_mmdg_pv_capacity), xytext=(-5, 5), textcoords='offset points', ha='right')
    plt.annotate('%.2f GW'%(current_mmdg_pv_capacity), (current_x, current_mmdg_pv_capacity), xytext=(0, -15), textcoords='offset points', ha='left')

    inflexion_date:str = '%02i/%04i'%(Z_ext_date[6350]%10000//100, Z_ext_date[6350]//10000)
    plt.plot(6350, pv_mmdg_installed_capacity(np.array([6350]))[0], marker='o', linestyle='-', color='black')
    plt.annotate(f'Inflexion MMDG PV Capacity (50% of Total Capacity)', (6350, pv_mmdg_installed_capacity(np.array([6350]))[0]), xytext=(-5, 5), textcoords='offset points', ha='right')
    plt.annotate('%.2f GW'%(pv_mmdg_installed_capacity(np.array([6350]))[0]), (6350, pv_mmdg_installed_capacity(np.array([6350]))[0]), xytext=(0, -15), textcoords='offset points', ha='left')

    plt.vlines(current_x, -5., 40., linestyles='dashed', color='black', alpha=0.4)

    plt.ylim((-5, 40))
    
    # Próximos passos:
    #   Agora que encontramos uma forma de linearizar a função a partir do períodos que de fato entrou em tendência,
    # devemos reconhecer que ele não seguira sendo uma power fanction ad infinitum.
    #   Logo, cabe determinar qual será o threshhold ou limite superior de capacidade instalada de energia
    # (muito provavelmente baseado na tendência de aumento de consumo elétrico, esse sim que tem uma tendência crescente entre linear ou pouco maior)
    # e o momento de inflexão dessa nova curva.
    # Analisar a tendência de consumo elétrico aqui: https://www.epe.gov.br/pt/areas-de-atuacao/energia-eletrica/consumo-de-energia-elétrica/painel-de-consumo-historico-de-energia-eletrica-desde-1970
    # Histórico capacidades instaladas SP: https://dadosabertos.aneel.gov.br/dataset/capacidade-instalada-por-unidade-da-federacao/resource/6fbee0f8-2617-4879-a69a-6b7892f12dad
    # Outro passo é entender essa nova curva chamada: Logistic Function (S-curve)
    # Olhar desmos com s-curve de teste

    """
    1. Qual a intenção dessa figura?
    4. Quando vc diz modelagem vc ta falando da estimação de GD instalada a partir dos dados da Aneel?
        Ser uma fonte de projeção da capacidade instalada de mmgd fotovoltaica no estado de São Paulo.

    2. O que significa todas as modalidades? PV, WIND, HYDRO ou todas as modalidades de GD(PV, PCH, CGH)?
        Os dados das capacidades totais instaladas em São Paulo foram tirados deste banco da ANEEL
        https://dadosabertos.aneel.gov.br/dataset/capacidade-instalada-por-unidade-da-federacao
        A partir deles, uma regressão polinomial foi realizada. Um polinômio de primeiro grau crescente foi o que melhor
        representou a constante e crescente capacidade total instalada no estado de São Paulo.
    
    3. Como vc chegou ao 70%?
    9. Qual o motivo da inflexão em 50%?
    
    5. O que significa o segundo ponto?
    
    6. A curva vermelha está dentro da azul?
    
    7. Se a curva vermelha foi criada a partir dos dados da ANEEL então temos que a geração por GD começou a aumentar a partir de 2017 o que faz sentido... 
    
    8. Acho que vale a pena traçar uma curva vertical tracejada indicando o HOJE ( fiquei com duvida se os 40% da capacidade instalada se referem ao primeiro ponto que aparentemente indica esse HOJE)
    
    """

    """ y:np.ndarray = np.arange(Z.shape[1])

    Y, X = np.meshgrid(y, x)

    Z_max_Y:np.ndarray = np.max(usedZ, axis=1)
    from scipy.ndimage import gaussian_filter1d
    smoothed_Z_max_Y = gaussian_filter1d(Z_max_Y, sigma=10)

    ax:Axes3D = plt.axes(projection='3d')

    ax.plot_surface(X, Y, usedZ, cmap='viridis')
    ax.plot(x, [y.max()]*len(x), smoothed_Z_max_Y, color='#440154', linewidth=2, label='Smoothed Z Max over X')
    ax.view_init(20, -50, 0) """

    all_idxs:list[int] = list(np.unique(Z_ext_date//1_0000, True)[1])
    idxs:list[int] = [all_idxs[i] for i in range(0, len(all_idxs), 2)]
    month_labels = ['%04i'%(Z_ext_date[i]//1_0000) for i in idxs]

    # Set the locations of the ticks to correspond to the first day of each unique month
    x_locator = FixedLocator(idxs)
    ax.xaxis.set_major_locator(x_locator)

    # Set the labels for these ticks using the month strings
    x_formatter = FixedFormatter(month_labels)
    ax.xaxis.set_major_formatter(x_formatter)

    ax.set_xlabel('Date [YYYY]')
    #ax.set_ylabel('Hour of the Day [Hour]')
    ax.set_ylabel('Power [GW]')
    ax.set_title('MMDG PV Capacity (%s/%s:%s/%s)\n[%s]\n\nCurrent Date: %s    Inflexion Date: %s'%(period[0][-2:], period[0][:-2], period[1][-2:], period[1][:-2], st, current_date, inflexion_date))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    #plt.savefig("%s\\outputs\\Ventures MMD PV Generation\\ventures-%s-(%i_%02i, %i_%02i).png"%(Path(dirname(abspath(__file__))).parent, state_abbreviation, period[0]//10000, period[0]%10000//100,  period[1]//10000, period[1]%10000//100), backend='Agg', dpi=200)
    plt.close()

if '__main__' == __name__:
    installed_power_analysis()