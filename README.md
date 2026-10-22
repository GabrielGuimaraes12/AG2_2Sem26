# AG2 - Classificação de canais de vendas

Projeto em Python que utiliza o conjunto de dados Wholesale customers para classificar uma venda como pertencente ao canal HoReCa ou Retail.

## Requisitos

- Python 3.11 ou superior
- Git

## Instalação

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-dev.txt
```

## Execução

```powershell
python main.py
```

O programa treina e avalia o modelo e, em seguida, solicita os campos Region, Fresh, Milk, Grocery, Frozen, Detergents Paper e Delicatessen para classificar uma nova amostra.

Os códigos de região são:

- `0`: Lisbon
- `1`: Oporto
- `2`: Other

## Preparação dos dados

O arquivo `data/wholesale.csv` contém 440 registros. Durante a preparação:

- Channel é convertido em HoReCa = 0 e Retail = 1;
- Region é convertido em Lisbon = 0, Oporto = 1 e Other = 2;
- as colunas são convertidas para `int64`;
- a variável alvo é separada dos sete atributos;
- os dados são divididos em 80% para treinamento e 20% para teste.

## Modelo

O classificador utilizado é `DecisionTreeClassifier` com `random_state=42`. A divisão usa estratificação para preservar a proporção das classes. O programa apresenta precision, recall, F1-score, suporte e acurácia.

## Estrutura

```text
data/          conjunto de dados
src/data.py    leitura e preparação
src/model.py   treinamento, avaliação e previsão
src/cli.py     entrada pelo terminal
tests/         testes automatizados
main.py        ponto de entrada
```

## Verificação

```powershell
python -m pytest -q
python -m ruff check .
python -m mypy src main.py tests
```
