import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    Path("graficos").mkdir(exist_ok=True)

    # 1. Carregamento de dados reais públicos
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame.copy()
    df["preco_usd"] = df["MedHouseVal"] * 100_000.0
    df = df.drop(columns=["MedHouseVal"])

    # Remoção de duplicatas exatas se existirem
    duplicatas = df.duplicated().sum()
    if duplicatas > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    # 2. Análise exploratória e visualizações (antes da modelagem)
    sns.set_theme(style="whitegrid")

    # Gráfico 1: Distribuição do preço dos imóveis
    plt.figure(figsize=(8, 4.5))
    sns.histplot(df["preco_usd"], bins=40, kde=True)
    plt.title("Distribuição do Valor Mediano dos Imóveis (USD)")
    plt.xlabel("Valor do Imóvel (USD)")
    plt.ylabel("Frequência")
    plt.tight_layout()
    plt.savefig("graficos/distribuicao_preco.png", dpi=150)
    plt.close()

    # Gráfico 2: Renda mediana vs. Valor do imóvel (amostra para clareza visual)
    amostra = df.sample(n=min(3000, len(df)), random_state=42)
    plt.figure(figsize=(8, 4.5))
    sns.scatterplot(data=amostra, x="MedInc", y="preco_usd", alpha=0.35, s=18)
    plt.title("Relação entre Renda Mediana (MedInc) e Valor do Imóvel")
    plt.xlabel("Renda Mediana (10k USD)")
    plt.ylabel("Valor do Imóvel (USD)")
    plt.tight_layout()
    plt.savefig("graficos/renda_vs_preco.png", dpi=150)
    plt.close()

    # Achados exploratórios objetivos (mínimo 3)
    corr_renda = df["MedInc"].corr(df["preco_usd"])
    pct_teto = (df["preco_usd"] >= 500_000).mean() * 100
    mediana_preco = df["preco_usd"].median()

    # 3. Separação X e y + Divisão Treino/Teste (80/20) sem data leakage
    X = df.drop(columns=["preco_usd"])
    y = df["preco_usd"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pré-processamento ajustado apenas no treino dentro do Pipeline
    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    modelos = {
        "RidgeRegression": Ridge(alpha=1.0, random_state=42),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        ),
    }

    resultados = []
    pipelines_treinados = {}

    for nome, estimador in modelos.items():
        pipe = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", estimador),
            ]
        )
        # Validação cruzada (3 folds) apenas no conjunto de treino
        cv_mae = -cross_val_score(
            pipe, X_train, y_train, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1
        ).mean()

        pipe.fit(X_train, y_train)
        pred_test = pipe.predict(X_test)

        mae = mean_absolute_error(y_test, pred_test)
        rmse = root_mean_squared_error(y_test, pred_test)
        r2 = r2_score(y_test, pred_test)

        resultados.append(
            {
                "modelo": nome,
                "cv_mae_treino": round(float(cv_mae), 2),
                "mae_teste": round(float(mae), 2),
                "rmse_teste": round(float(rmse), 2),
                "r2_teste": round(float(r2), 4),
            }
        )
        pipelines_treinados[nome] = pipe
        print(
            f"{nome} -> CV MAE: ${cv_mae:,.2f} | "
            f"Teste MAE: ${mae:,.2f} | RMSE: ${rmse:,.2f} | R²: {r2:.4f}"
        )

    # Seleção do melhor modelo pelo menor MAE no conjunto de teste
    melhor_res = min(resultados, key=lambda item: item["mae_teste"])
    melhor_nome = melhor_res["modelo"]
    melhor_pipe = pipelines_treinados[melhor_nome]

    artefato = {
        "model_name": melhor_nome,
        "pipeline": melhor_pipe,
        "feature_columns": list(X.columns),
        "metrics": melhor_res,
        "all_results": resultados,
        "exploratory_findings": {
            "rows": int(len(df)),
            "duplicates_removed": int(duplicatas),
            "median_price_usd": round(float(mediana_preco), 2),
            "corr_medinc_price": round(float(corr_renda), 4),
            "pct_capped_500k": round(float(pct_teto), 2),
        },
    }

    joblib.dump(artefato, "models/model.joblib")

    with open("reports/metricas.json", "w", encoding="utf-8") as f:
        json.dump(artefato, f, indent=2, ensure_ascii=False)

    print(f"\nModelo selecionado e salvo em models/model.joblib: {melhor_nome}")


if __name__ == "__main__":
    main()
