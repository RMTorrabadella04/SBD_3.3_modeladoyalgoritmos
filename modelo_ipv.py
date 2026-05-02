import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from preparacion_datasets import *

import matplotlib.pyplot as plt

def ipv():
    df = dataframe_ipv()

    FEATURES = [
        "t",
        "year",
        "quarter",
        "q_sin", "q_cos",
        "lag1", "lag2", "lag3", "lag4", "lag8",
        "roll2", "roll4",
        "diff1", "diff4",
        "ewm2", "ewm4"
    ]

    X = df[FEATURES]

    # TARGET
    y = np.log1p(df["Valor"])

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.02,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    tscv = TimeSeriesSplit(n_splits=5)

    rmses = []
    maes = []
    r2s = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        # revertir log
        y_val_real = np.expm1(y_val)
        y_pred_real = np.expm1(y_pred)

        rmse = np.sqrt(mean_squared_error(y_val_real, y_pred_real))
        mae = mean_absolute_error(y_val_real, y_pred_real)
        r2 = r2_score(y_val_real, y_pred_real)

        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)

        print(f"Fold {fold+1} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

    print("\nRESULTADOS FINALES")
    print(f"RMSE medio: {np.mean(rmses):.4f}")
    print(f"MAE medio: {np.mean(maes):.4f}")
    print(f"Último RMSE (real): {rmses[-1]:.4f}")
    print(f"Último RMSE (real): {maes[-1]:.4f}")
    print(f"R2: {r2s[-1]:.4f}")
    
    
    model.fit(X, y)
    pred = np.expm1(model.predict(X))
    real = df["Valor"].values

    plt.figure()
    plt.plot(real, label="Real")
    plt.plot(pred, label="Predicción")
    plt.title("IPV - Real vs Predicción")
    plt.legend()
    plt.show()