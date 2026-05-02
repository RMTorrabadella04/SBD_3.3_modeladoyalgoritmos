import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from sklearn.metrics import r2_score
from preparacion_datasets import *

import matplotlib.pyplot as plt


def ipc():
    df = dataframe_ipc()

    FEATURES = [
        "t",
        "mes_sin", "mes_cos",
        "quarter",
        "lag1", "lag2", "lag3", "lag6", "lag12",
        "roll3", "roll6",
        "diff1", "diff12",
        "ewm3", "ewm6"
    ]

    X = df[FEATURES]

    # log seguro
    y = np.log1p(df["Valor"])

    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.02,
        max_depth=3,
        min_child_weight=3,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42
    )

    tscv = TimeSeriesSplit(n_splits=5)

    rmses = []
    maes = []
    r2s = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Compatible con todas las versiones
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        # Revertir log
        y_val_exp = np.expm1(y_val)
        y_pred_exp = np.expm1(y_pred)

        rmse = np.sqrt(mean_squared_error(y_val_exp, y_pred_exp))
        mae = mean_absolute_error(y_val_exp, y_pred_exp)
        r2 = r2_score(y_val_exp, y_pred_exp)

        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)

        print(f"Fold {fold+1} RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

    print("\nRESULTADOS FINALES:")
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
    plt.title("IPC - Real vs Predicción")
    plt.legend()
    plt.show()