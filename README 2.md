# Fonte de Dados — California Housing Dataset

- **Origem:** `sklearn.datasets.fetch_california_housing` (base clássica derivada do Censo da Califórnia de 1990, disponibilizada pelo scikit-learn).
- **Licença / Condições de uso:** Disponível para fins académicos, de pesquisa e demonstração via scikit-learn (dados censitários agregados; sem dados pessoais identificáveis).
- **Variável-alvo (`target`):** Valor mediano das casas por quarteirão/região, originalmente expresso em centenas de milhares de dólares (`MedHouseVal`). No `train.py`, convertida para dólares (`valor_usd = MedHouseVal * 100_000`) para facilitar a leitura.

## Atributos (Features)
- `MedInc`: Renda mediana do grupo de domicílios (em dezenas de milhares de USD).
- `HouseAge`: Idade mediana dos imóveis na região (em anos).
- `AveRooms`: Média de cômodos por domicílio.
- `AveBedrms`: Média de quartos por domicílio.
- `Population`: População da região.
- `AveOccup`: Média de moradores por domicílio.
- `Latitude`: Latitude geográfica.
- `Longitude`: Longitude geográfica.
