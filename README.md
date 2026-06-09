# World Cup Pool — Copa 2026

Previsão dos placares da **1ª rodada da fase de grupos** da Copa do Mundo FIFA
2026 com uma rede neural em **TensorFlow/Keras**.

Projeto em **scripts Python** (sem Jupyter), com repositório Git reprodutível.

> **Hipótese:** a força relativa das seleções (Elo + ranking FIFA + forma
> recente + contexto do jogo) permite estimar a distribuição de gols e, via rede
> neural, convertê-la num par de gols inteiros por partida.

---

## Requisitos e instalação

- **Python 3.10–3.13** (TensorFlow ainda não suporta 3.14 no Windows)

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Como executar

Na raiz do projeto, com o ambiente ativado:

```bash
python main.py                  # pipeline completo: dados → features → treino → avaliação → JSON
python main.py --force-download # idem, rebaixando os CSVs

# ou módulo a módulo:
python -m src.data        # download + resumo dos dados
python -m src.features    # gera features de treino e inferência
python -m src.model       # treina e salva o modelo
python -m src.evaluate    # pontuação do bolão vs baselines
python -m src.predict     # gera o bolao_copa.txt
```

A reprodutibilidade é garantida pela semente fixa `RANDOM_SEED = 42`
(`src/config.py`), aplicada a `python`, `numpy` e `tensorflow`.

## Estrutura do projeto

```
world-cup-pool/
├── main.py                 # pipeline ponta a ponta (Fases 2–6)
├── bolao_copa.txt          # JSON final de entrega
├── requirements.txt
├── data/
│   ├── raw/                # CSVs baixados + sources.json
│   └── processed/          # features, medianas e scaler
├── outputs/                # model.keras, histórico, avaliação, previsões
└── src/
    ├── config.py           # constantes, siglas, sementes
    ├── data.py             # download e carga dos dados (Fase 2)
    ├── team_names.py       # mapeamento nome → sigla oficial
    ├── features.py         # engenharia de features (Fase 3)
    ├── model.py            # MLP TensorFlow/Keras (Fase 4)
    ├── evaluate.py         # pontuação do bolão e baselines (Fase 5)
    └── predict.py          # previsões e exportação do JSON (Fase 6)
```

---

## 1. Introdução

O objetivo é prever o placar dos **24 primeiros jogos** da fase de grupos da Copa
2026 e entregar as previsões no formato `bolao_copa.txt`. A pontuação do bolão
premia **5 pontos** por placar exato e **2 pontos** por acertar apenas o
resultado (vitória/empate/derrota), totalizando até 120 pontos.

A abordagem é uma **regressão de gols** com rede neural: em vez de só classificar
o vencedor, o modelo estima quantos gols cada seleção marca, o que permite mirar
o placar exato.

## 2. Dados

Três fontes públicas, todas com corte **anterior a 10/06/2026** (sem usar dados
futuros). Metadados gravados em `data/raw/sources.json`.

| Arquivo | Fonte | O que traz |
|---------|-------|------------|
| `international_results.csv` | [martj42/international_results](https://github.com/martj42/international_results) | Histórico de partidas entre seleções (gols, torneio, mando) |
| `fifa_ranking_historical.csv` | [Dato-Futbol/fifa-ranking](https://github.com/Dato-Futbol/fifa-ranking) | Ranking FIFA histórico (pontos) |
| `elo_ratings_yearly.csv` | [JGravier/soccer-elo](https://github.com/JGravier/soccer-elo) | Elo anual das seleções (eloratings.net) |

O `international_results` fornece os jogos de treino e a forma recente; FIFA e Elo
medem a força relativa das seleções antes de cada partida.

## 3. Pré-processamento (features)

`src/features.py` transforma os dados brutos em uma tabela por partida, em
**espaço de siglas** e **sem vazamento temporal** — cada feature usa apenas dados
anteriores ao jogo.

**Codificação dos países.** Os três datasets usam grafias inglesas divergentes
(ex.: "Côte d'Ivoire" vs "Ivory Coast"). A resolução une o código `team_short` do
FIFA (autoritativo) com o `NAME_VARIANTS` curado, cobrindo ~205 seleções e
mapeando as 48 da Copa 2026 para as siglas oficiais do enunciado.

**Features (16 colunas):**

| Feature | Descrição |
|---------|-----------|
| `elo_a`, `elo_b`, `elo_diff` | Elo do ano anterior à partida |
| `fifa_a`, `fifa_b`, `fifa_diff` | Pontos FIFA mais recentes *antes* do jogo (junção as-of) |
| `form_gf_*`, `form_ga_*`, `form_ppg_*` | Forma nas últimas 10 partidas: gols pró, gols contra e pontos por jogo (com `shift` para excluir o próprio jogo) |
| `neutral` | 1 em campo neutro (anfitriões USA/MEX/CAN jogam como casa) |

**Alvos:** `gols_a` (casa/listado) e `gols_b` (visitante), inteiros.

**Normalização.** Imputação por mediana do treino (`feature_medians.json`) e
padronização *StandardScaler* com média/desvio do treino (`feature_scaler.json`),
ambos reaproveitados na inferência para evitar vazamento.

Saídas em `data/processed/`: `train_features.csv` (~26,9 mil partidas desde 1994),
`inference_features.csv` (24 jogos de 2026), `feature_medians.json`,
`feature_scaler.json`.

## 4. Modelo TensorFlow/Keras

`src/model.py` treina um **MLP** que recebe as 16 features padronizadas e prevê o
par de gols `(casa, visitante)`:

```
Input(16) → Dense(128, relu) → Dropout(0.2)
          → Dense(64, relu)  → Dropout(0.2)
          → Dense(32, relu)  → Dense(2)
```

- **Split temporal** (não aleatório): treino `< 2018`, validação `2018–2021`,
  teste `>= 2022` — nunca usa o futuro para prever o passado.
- Perda `MSE`, otimizador `Adam` (lr 1e-3), `EarlyStopping` na validação,
  semente fixa.
- Pós-processamento (Fase 6): arredondar para inteiros e `clip(min=0)`.

Saídas em `outputs/`: `model.keras` e `training_history.json`
(MAE ≈ 0,9 gol em validação e teste, sem overfit).

## 5. Avaliação

`src/evaluate.py` simula a **pontuação do bolão** no conjunto de teste (jogos a
partir de 2022, nunca vistos pelo modelo) e compara com três baselines.

| Estratégia | pts/jogo | placar exato | resultado | MAE |
|------------|:--------:|:------------:|:---------:|:---:|
| **Modelo (MLP)** | **1,42** | **11,7 %** | 53,2 % | **0,88** |
| Maior Elo 2×1 | 1,39 | 7,9 % | 57,7 % | 0,98 |
| Placar médio | 1,16 | 7,3 % | 47,3 % | 1,05 |
| Empate 1×1 | 0,79 | 10,6 % | 23,7 % | 0,99 |

O modelo supera todos os baselines na métrica que vale (pts/jogo) e crava ~50 %
mais placares exatos que o melhor baseline. Resultado salvo em
`outputs/evaluation.json`. Valores reprodutíveis (modo determinístico do TF +
semente fixa); `python main.py` gera exatamente o mesmo `bolao_copa.txt` a cada
execução.

## 6. Previsões + JSON

`src/predict.py` carrega o modelo treinado, prevê os 24 jogos a partir das
features de inferência, arredonda para gols inteiros ≥ 0 e grava o
`bolao_copa.txt` no formato exigido (ordem oficial `jogo1`…`jogo24`, siglas e
chaves na ordem da tabela). Uma cópia validada vai para `outputs/previsoes.json`.

Formato de cada jogo:

```json
"jogo1": { "MEX": { "gols": 2 }, "RSA": { "gols": 1 } }
```

## 7. Conclusão

O pipeline cobre da coleta de dados à exportação do JSON com um único comando
(`python main.py`) e é reprodutível por semente fixa. A rede neural, treinada com
validação temporal e sem vazamento, **supera os baselines** na pontuação do bolão,
principalmente por acertar mais placares exatos — exatamente onde estão os 5
pontos por jogo.

**Limitações e melhorias futuras:** features de confederação/sede e fase do
torneio foram omitidas por falta de cobertura histórica confiável; uma cabeça de
saída Poisson (gols são contagens) e embeddings por seleção poderiam refinar as
estimativas. Ainda assim, a abordagem entrega previsões coerentes e justificáveis
para as 48 seleções.

---

### Checklist de entrega

- [x] Modelo de rede neural em TensorFlow/Keras
- [x] Repositório reprodutível (semente fixa, dependências em `requirements.txt`)
- [x] Dados, fontes e tratamento documentados
- [x] Split treino/validação/teste (temporal)
- [x] `bolao_copa.txt` com `nome`, `turma` e os 24 jogos no formato correto
