import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def dataframe_ipc():
    df_ipc = pd.read_csv('data/ipc_Data.csv', sep=";")
    df_ipcp = pd.read_csv('data/ipc_principal.csv', sep=";")
    
    df_ipc_final = df_ipc
    
    mapeo = df_ipcp.set_index("COD")["Nombre"]
    
    df_ipc_final["Nombre_Categoria"] = df_ipc_final["REF_ID_RELACIONAL"].map(mapeo)
    
    df_ipc_final = df_ipc_final.rename(columns={'REF_ID_RELACIONAL': 'COD'})
    
    # Convertimos a float los Strings de los valores a Normalizar
    
    df_ipc_final["Valor"] = df_ipc_final["Valor"].astype(str).str.replace(",", ".").astype(float)
    
    df_ipc_final["Promedio_Trimestral"] = df_ipc_final["Promedio_Trimestral"].astype(str).str.replace(",", ".").astype(float)
    
    df_ipc_final["Fecha"] = pd.to_datetime(df_ipc_final["Fecha"], dayfirst=True)
    
    # Normalizamos los datos

    scaler = StandardScaler()
    
    columnas_standard = ["Valor", "Promedio_Trimestral"]
    
    minmaxscaler = MinMaxScaler()
    
    columnas_minmax = ["Trimestre"]
    
    df_ipc_final[columnas_standard] = scaler.fit_transform(df_ipc_final[columnas_standard])
    
    df_ipc_final[columnas_minmax] = minmaxscaler.fit_transform(df_ipc_final[columnas_minmax])
    
    return df_ipc_final


def dataframe_ipv():
    df_ipv = pd.read_csv('data/ipv_Data.csv', sep=";")
    df_ipvp = pd.read_csv('data/ipv_principal.csv', sep=";")
    
    df_ipv_final = df_ipv
    
    mapeo = df_ipvp.set_index("COD")["Nombre"]

    df_ipv_final["Nombre_Categoria"] = df_ipv_final["REF_ID_RELACIONAL"].map(mapeo)

    # Convertimos a float los Strings de los valores a Normalizar

    df_ipv_final["Valor"] = df_ipv_final["Valor"].astype(str).str.replace(",", ".").astype(float)
    
    df_ipv_final["Variacion_Porcentual_Trimestre"] = df_ipv_final["Variacion_Porcentual_Trimestre"].astype(str).str.replace(",", ".").astype(float)
    
    df_ipv_final["Fecha"] = pd.to_datetime(df_ipv_final["Fecha"], dayfirst=True)
    
    # Tenemos valores que tienden al Infito por lo que los borraremos
    
    df_ipv_final["Variacion_Porcentual_Trimestre"] = df_ipv_final["Variacion_Porcentual_Trimestre"].replace([np.inf, -np.inf], np.nan)

    df_ipv_final = df_ipv_final.dropna(subset=["Variacion_Porcentual_Trimestre"])
    
    df_ipv_final = df_ipv_final.rename(columns={'REF_ID_RELACIONAL': 'COD'})
    
    # Normalizamos los datos
    
    scaler = StandardScaler()
    
    columnas_standard = ["Variacion_Porcentual_Trimestre"]
    
    minmaxscaler = MinMaxScaler()
    
    columnas_minmax = ["Valor", "Trimestre"]
    
    df_ipv_final[columnas_standard] = scaler.fit_transform(df_ipv_final[columnas_standard])
    
    df_ipv_final[columnas_minmax] = minmaxscaler.fit_transform(df_ipv_final[columnas_minmax])
    
    return df_ipv_final
