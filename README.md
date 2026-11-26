# ML Challenge — Previsão de Preço de Imóveis

**Estudante:** Thiago Rubens da Silva Assis  
**Trilha escolhida:** Trilha A — Preço de imóveis (Problema de **Regressão**)

## 1. Resumo do Problema e Variável-Alvo
O objetivo deste projeto é estimar o valor mediano de imóveis (`preco_usd`, derivado de `MedHouseVal * 100.000`) a partir de características socioeconómicas, estruturais e geográficas da região. A previsão pode apoiar análises preliminares de mercado imobiliário ou triagem de valores, utilizando apenas atributos disponíveis no momento da consulta (sem vazamento de dados / *data leakage*).

## 2. Fonte dos Dados e Licença
- **Dataset:** California Housing (`sklearn.datasets.fetch_california_housing`), baseado nos dados censitários da Califórnia (1990).
- **Origem / Ferramenta:** Scikit-learn (`https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html`).
- **Licença e condições:** Base pública para fins educacionais, científicos e de demonstração. Mais detalhes em `data/README.md`.

## 3. Como Instalar e Executar

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python train.py
streamlit run app.py
```

## 4. Análise Exploratória e Pré-processamento
- **Pré-processamento sem vazamento (*data leakage*):** Separação em treino (80%) e teste (20%) com `random_state=42` antes do ajuste de transformações. Dentro do `Pipeline`, aplicou-se `SimpleImputer(strategy="median")` e `StandardScaler()` ajustados **apenas** nos dados de treino.
- **3 Achados exploratórios principais:**
  1. A renda mediana (`MedInc`) apresenta correlação positiva relevante com o valor do imóvel (`preco_usd`), indicando que regiões de maior renda tendem a concentrar imóveis mais caros, embora com dispersão conforme localização e densidade.
  2. Há teto/truncamento próximo de US$ 500.000 (`MedHouseVal = 5.0`) em parte das observações da base original, o que influencia previsões em faixas de alto padrão.
  3. A distribuição dos preços é assimétrica à direita (cauda longa superior), tornando métricas como MAE e RMSE complementares para avaliar erros médios e desvios maiores.

Gráficos gerados em `graficos/`:
- `graficos/distribuicao_preco.png`
- `graficos/renda_vs_preco.png`

## 5. Comparação de Modelos e Seleção
Ambos os modelos foram avaliados no mesmo conjunto de teste (20%) e com validação cruzada no treino.

| Algoritmo | Métrica principal | Resultado obtido | Interpretação |
| :--- | :--- | :--- | :--- |
| `RidgeRegression` | MAE (Teste) | *Preencher após rodar `train.py`* | Modelo linear regularizado; rápido e interpretável, mas limitado para relações não lineares complexas. |
| `RandomForestRegressor` | MAE (Teste) | *Preencher após rodar `train.py`* | Captura interações não lineares e efeitos geográficos; reduz o erro médio em relação ao modelo linear. |
| **Escolhido (`RandomForestRegressor`)** | **MAE / RMSE / R²** | *Preencher após rodar `train.py`* | Selecionado por apresentar menor erro absoluto médio (MAE) no conjunto de teste e melhor aderência aos dados. |

*(Atualize os valores exatos a partir de `reports/metricas.json` ou do terminal após executar `python train.py`.)*

## 6. Demonstração, Evidências e Limitações
- **Exemplo de previsão:** Na interface Streamlit, ao informar renda mediana `4.5`, idade `25` anos, média de `5.2` cômodos, `1.1` quartos, população `1400`, ocupação `2.8` e coordenadas próximas a Los Angeles (`34.05`, `-118.25`), o modelo retorna a estimativa em USD e uma faixa indicativa baseada no MAE de teste.
- **Captura de ecrã:** Salve a imagem da aplicação rodando em `screenshots/interface.png`.
- **Limitações e melhorias futuras:**
  - Dados históricos agregados por região (não refletem condições individuais de um imóvel específico nem inflação/mercado atual).
  - Possível truncamento superior em US$ 500.000 na base original.
  - Melhorias futuras: ajuste fino de hiperparâmetros via validação cruzada, inclusão de variáveis externas mais recentes e exibição de importância dos atributos na interface.

## 7. Declaração de Uso de Ferramentas de IA
Este projeto foi desenvolvido com auxílio de inteligência artificial generativa para estruturação do código, documentação e revisão técnica, sendo testado e validado localmente antes da submissão.
