import pandas as pd
import numpy as np

def dataframe_ipc():
    df = pd.read_csv('data/ipc_Data.csv', sep=";")

    # eliminar filas corruptas del header repetido
    df = df[df["Valor"].astype(str) != "Valor"]

    # limpieza segura
    df["Valor"] = (
        df["Valor"]
        .astype(str)
        .str.replace(",", ".", regex=False)
    )

    df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")

    df["Fecha"] = pd.to_datetime(df["Fecha"])

    df = df.sort_values("Fecha").reset_index(drop=True)

    # limpiar nulos / infinitos
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Valor"])
    df = df[df["Valor"] > 0]

    # Features temporales
    df["Mes"] = df["Fecha"].dt.month
    df["quarter"] = df["Fecha"].dt.quarter

    df["mes_sin"] = np.sin(2 * np.pi * df["Mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["Mes"] / 12)

    df["t"] = np.arange(len(df))

    # LAGS
    for lag in [1, 2, 3, 6, 12]:
        df[f"lag{lag}"] = df["Valor"].shift(lag)

    # Rolling
    df["roll3"] = df["Valor"].shift(1).rolling(3).mean()
    df["roll6"] = df["Valor"].shift(1).rolling(6).mean()

    # Diferencias
    df["diff1"] = df["Valor"].diff(1)
    df["diff12"] = df["Valor"].diff(12)

    # EWM
    df["ewm3"] = df["Valor"].shift(1).ewm(span=3).mean()
    df["ewm6"] = df["Valor"].shift(1).ewm(span=6).mean()

    df = df.dropna().reset_index(drop=True)

    return df


def dataframe_ipv():
    df = pd.read_csv("data/ipc_Data.csv", sep=";")

    # limpiar números con coma decimal (robusto)
    for col in ["Valor", "Variacion_Porcentual_Trimestre"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", "."),
                errors="coerce"
            )

    # eliminar filas corruptas
    df = df.dropna(subset=["Valor"]).reset_index(drop=True)

    # Fecha
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"])
    df = df.sort_values("Fecha").reset_index(drop=True)

    # eliminar infinitos
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Valor"])

    # features temporales (TRIMESTRAL)
    df["year"] = df["Fecha"].dt.year
    df["quarter"] = df["Fecha"].dt.quarter

    # ciclo trimestral
    df["q_sin"] = np.sin(2 * np.pi * df["quarter"] / 4)
    df["q_cos"] = np.cos(2 * np.pi * df["quarter"] / 4)

    df["t"] = np.arange(len(df))

    # LAGS
    for lag in [1, 2, 3, 4, 8]:
        df[f"lag{lag}"] = df["Valor"].shift(lag)

    # rolling (sin leakage)
    df["roll2"] = df["Valor"].shift(1).rolling(2).mean()
    df["roll4"] = df["Valor"].shift(1).rolling(4).mean()

    # diferencias
    df["diff1"] = df["Valor"].diff(1)
    df["diff4"] = df["Valor"].diff(4)

    # ewm
    df["ewm2"] = df["Valor"].shift(1).ewm(span=2).mean()
    df["ewm4"] = df["Valor"].shift(1).ewm(span=4).mean()

    # limpiar NaNs generados por lags/rolling
    df = df.dropna().reset_index(drop=True)

    return df