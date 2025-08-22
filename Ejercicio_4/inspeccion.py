import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import load_data
from scipy.signal import find_peaks

def time_infection(dis, continuous):
    # Prepare data
    print(dis["timestamps"])
    print(dis["infections"])
    
    plt.plot(
        dis["timestamps"][:len(dis["infections"])], 
        dis["infections"], label='Discreto')
    plt.plot(con["timestamps"], con["infections"], label='Continuo')
    plt.title("Distribucion temporal de poblacion infectada")
    plt.xlabel("Tiempo (dia)")
    plt.ylabel("Infectados")
    plt.legend()
    plt.tight_layout()
    plt.show()

def feature_histogram(dis, con):
    features = dis["agent_data"].keys()
    agent_data = pd.DataFrame(dis["agent_data"])
    fig, axes = plt.subplots(1, 3, figsize=(8,4))
    fig.suptitle("Rasgos de agentes: temporal Discreta")
    for i,f in enumerate(features):   
        val_counts = agent_data[f].value_counts()
        axes[i].bar(val_counts.index, val_counts.values)
        axes[i].set_xticklabels(val_counts.index,ha='right', rotation=30)
        axes[i].set_title(f"Histograma \'{f}\'")
        axes[i].set_xlabel(f)
        axes[i].set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
    features = con["agent_data"].keys()
    agent_data = pd.DataFrame(con["agent_data"])
    fig, axes = plt.subplots(1, 3, figsize=(8,4))
    fig.suptitle("Rasgos de agentes: temporal Continua")
    for i,f in enumerate(features):  
        if(agent_data[f].dtype ==float):
            axes[i].hist(agent_data[f])
            axes[i].set_xticklabels(val_counts.index,ha='right', rotation=30)
            axes[i].set_title(f"Histograma \'{f}\'")
            axes[i].set_xlabel(f"{f} - Continuous")
            axes[i].set_ylabel("Frequency")  
        else:
            val_counts = agent_data[f].value_counts()
            axes[i].bar(val_counts.index, val_counts.values)
            axes[i].set_xticklabels(val_counts.index,ha='right', rotation=30)
            axes[i].set_title(f"DHistograma \'{f}\'")
            axes[i].set_xlabel(f"{f} - Discrete")
            axes[i].set_ylabel("Frequency")   
        
    plt.tight_layout()
    plt.show()

def analyze(dis, con):
    time_infection(dis, con)
    feature_histogram(dis, con)

def time_peaks_con(con):
    infected = pd.DataFrame({"infected": con["infections"]}, index=con["timestamps"])
    cases = infected["infected"].values
    peaks_idx, properties = find_peaks(cases, height=80, prominence=5)
    peaks_df = pd.DataFrame({
        "time": infected.index[peaks_idx],
        "cases": cases[peaks_idx]
    })

    # plt.title("Distribucion temporal Continua - Picos")
    # plt.plot(infected)
    # plt.scatter(peaks_df["time"], peaks_df["cases"], color='r', marker='o')
    # plt.xlabel("Tiempo")
    # plt.ylabel("Infectados")
    # plt.tight_layout()
    # plt.show()
    
    # plt.title("Diferencial Temporal - Continua")
    # plt.scatter(peaks_df.index ,peaks_df["time"].diff(), label='Diferenciales')
    # plt.xlabel("Indice")
    # plt.ylabel("Diferencial")
    # mean = peaks_df["time"].diff().mean()
    # plt.axhline(mean, linestyle='--', color='r', label='Media')
    # plt.show()

    print("Low Level Transmission")
    for i in range(len(peaks_df)-1):
        start = peaks_df["time"].iloc[i]
        end = peaks_df["time"].iloc[i+1]
        low_transmission = infected[(infected.index > start) & (infected.index < end)]
        sum  = low_transmission.sum().tolist()[0]
        print(f"Between {start} and {end}, low-level counts: {sum}")

def time_peaks_dis(dis):
    infected = pd.DataFrame({"infected": dis["infections"]}, index=dis["timestamps"][:len(dis["infections"])])
    peaks = infected[infected["infected"]> 30]
    
    plt.title("Distribucion temporal Discreta - Picos")
    plt.plot(infected)
    plt.scatter(peaks.index, peaks["infected"], color='r', marker='o')
    plt.xlabel("Tiempo")
    plt.ylabel("Infectados")
    plt.tight_layout()
    plt.show()
    
    plt.title("Diferencial Temporal - Discreta")
    plt.scatter(peaks.index, peaks.index.diff(), label='Diferenciales')
    plt.xlabel("Indice")
    plt.ylabel("Diferencial")
    mean = peaks.index.to_series().diff().mean()
    plt.axhline(mean, linestyle='--', color='r', label='Media')
    plt.show()
    
    indexes = peaks.index.tolist()
    print("Low Level Transmission")
    for i in range(len(indexes)-1):
        start = indexes[i]
        end = indexes[i+1]
        low_transmission = infected[(infected.index > start) & (infected.index < end)]
        sum  = low_transmission.sum().tolist()[0]
        print(f"Between {start} and {end}, low-level counts: {sum}")

    

if __name__ == "__main__":
    data = load_data.load_data()
    dis = data["discrete"]
    con = data["continuous"]
    # analyze(dis, con)
    # time_peaks_dis(dis)
    time_peaks_con(con)