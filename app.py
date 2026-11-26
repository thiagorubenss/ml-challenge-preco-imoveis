from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ML Challenge — Previsão de Preço de Imóveis",
    page_icon="🏠",
    layout="centered",
)

st.title("🏠 Estimativa de Valor de Imóveis (Regressão)")
st.caption(
    "Aviso: Este é um protótipo educacional de Machine Learning. "
    "As estimativas são aproximações estatísticas sujeitas a erros e não substituem avaliação imobiliária profissional."
)

st.markdown(
    "Preencha os atributos regionais e estruturais do imóvel abaixo para gerar uma previsão "
    "utilizando o pipeline treinado salvo em disco."
)

MODEL_PATH = Path("models/model.joblib")

if not MODEL_PATH.exists():
    st.error(
        "Arquivo do modelo não encontrado em `models/model.joblib`. "
        "Execute primeiro `python train.py` no terminal para treinar e salvar o modelo."
    )
    st.stop()

try:
    artefato = joblib.load(MODEL_PATH)
    pipeline = artefato["pipeline"]
    model_name = artefato.get("model_name", "Modelo de Regressão")
    metrics = artefato.get("metrics", {})
except Exception as exc:
    st.error(f"Falha ao carregar o modelo salvo: {exc}")
    st.stop()

with st.expander("Informações do modelo carregado"):
    st.write(f"**Algoritmo selecionado:** `{model_name}`")
    if metrics:
        st.write(f"**MAE (erro absoluto médio no teste):** US$ {metrics.get('mae_teste', 0):,.2f}")
        st.write(f"**RMSE no teste:** US$ {metrics.get('rmse_teste', 0):,.2f}")
        st.write(f"**R² no teste:** {metrics.get('r2_teste', 0):.4f}")

with st.form("form_previsao"):
    col1, col2 = st.columns(2)

    with col1:
        med_inc = st.number_input(
            "Renda Mediana (em dezenas de mil USD)",
            min_value=0.5,
            max_value=15.0,
            value=4.5,
            step=0.1,
            help="Ex.: 4.5 representa US$ 45.000 de renda mediana na região.",
        )
        house_age = st.number_input(
            "Idade Mediana dos Imóveis (anos)",
            min_value=1.0,
            max_value=60.0,
            value=25.0,
            step=1.0,
        )
        ave_rooms = st.number_input(
            "Média de Cômodos por Domicílio",
            min_value=1.0,
            max_value=20.0,
            value=5.2,
            step=0.1,
        )
        ave_bedrms = st.number_input(
            "Média de Quartos por Domicílio",
            min_value=0.5,
            max_value=10.0,
            value=1.1,
            step=0.1,
        )

    with col2:
        population = st.number_input(
            "População da Região",
            min_value=10.0,
            max_value=40000.0,
            value=1400.0,
            step=50.0,
        )
        ave_occup = st.number_input(
            "Média de Moradores por Domicílio",
            min_value=0.5,
            max_value=20.0,
            value=2.8,
            step=0.1,
        )
        latitude = st.number_input(
            "Latitude (Califórnia)",
            min_value=32.0,
            max_value=42.5,
            value=34.05,
            step=0.05,
        )
        longitude = st.number_input(
            "Longitude (Califórnia)",
            min_value=-125.0,
            max_value=-114.0,
            value=-118.25,
            step=0.05,
        )

    submitted = st.form_submit_button("Fazer previsão", type="primary")

if submitted:
    # Validação básica de coerência
    if ave_bedrms > ave_rooms:
        st.warning(
            "Atenção: a média de quartos informada é maior que a média total de cômodos. "
            "Verifique os valores para uma entrada mais realista."
        )

    entrada_df = pd.DataFrame(
        [
            {
                "MedInc": float(med_inc),
                "HouseAge": float(house_age),
                "AveRooms": float(ave_rooms),
                "AveBedrms": float(ave_bedrms),
                "Population": float(population),
                "AveOccup": float(ave_occup),
                "Latitude": float(latitude),
                "Longitude": float(longitude),
            }
        ]
    )

    try:
        previsao = float(pipeline.predict(entrada_df)[0])
        previsao_ajustada = max(0.0, previsao)

        st.success(f"**Valor estimado do imóvel:** US$ {previsao_ajustada:,.2f}")

        mae_ref = metrics.get("mae_teste")
        if mae_ref is not None:
            faixa_min = max(0.0, previsao_ajustada - mae_ref)
            faixa_max = previsao_ajustada + mae_ref
            st.info(
                f"**Explicação do resultado:** O modelo **{model_name}** estimou o valor com base "
                f"nos padrões aprendidos no conjunto de treinamento. Considerando o erro absoluto "
                f"médio (MAE) de **US$ {mae_ref:,.2f}** no teste, uma faixa indicativa de "
                f"referência fica entre **US$ {faixa_min:,.2f}** e **US$ {faixa_max:,.2f}**."
            )
    except Exception as exc:
        st.error(f"Erro ao realizar a previsão: {exc}")
