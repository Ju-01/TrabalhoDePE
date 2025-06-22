# Análise Descritiva e Regressão Linear - Fatores de Risco de AVC

O Acidente Vascular Cerebral (AVC) é uma das principais causas de mortalidade e incapacitação no mundo. Diversos estudos apontam fatores de risco bem estabelecidos para o AVC, incluindo hipertensão arterial, diabetes, doenças cardíacas, tabagismo e idade avançada.

Por exemplo, estima-se que cerca de 70% dos pacientes que sofrem AVC apresentavam hipertensão, e o AVC chega a ser quatro vezes mais comum em indivíduos diabéticos em comparação à população geral. Além disso, o hábito de fumar aumenta significativamente a probabilidade de AVC, e o risco de derrame dobra a cada década após os 55 anos de idade.

Conhecer o perfil desses fatores de risco na população pode ajudar na prevenção e no desenvolvimento de políticas de saúde. Neste trabalho, analisaremos um dataset público de pacientes que reúne informações demográficas e de saúde relacionadas a esses fatores de risco, bem como a ocorrência (ou não) de AVC para cada paciente. Este conjunto de dados foi obtido no contexto de uma competição do Kaggle e contém diversas variáveis relevantes para prever a chance de AVC a partir dos fatores mencionados.

Cada linha do dataset representa um paciente, incluindo se ele teve AVC (stroke = 1 para sim, 0 para não) e as seguintes características:

* **gender**: gênero biológico do paciente (Male, Female ou Other);
* **age**: idade do paciente (em anos);
* **hypertension**: histórico de hipertensão (1 = sim, 0 = não);
* **heart_disease**: presença de doença cardíaca (1 = sim, 0 = não);
* **ever_married**: estado civil (se o paciente já foi casado – Yes ou No);
* **work_type**: tipo de trabalho (Private = setor privado, Self-employed = autônomo, Govt_job = servidor público, children = crianças, Never_worked = nunca trabalhou);
* **Residence_type**: tipo de residência (Urban = urbana, Rural = rural);
* **avg_glucose_level**: nível médio de glicose no sangue do paciente (mg/dL);
* **bmi**: Índice de Massa Corporal do paciente (kg/m²);
* **smoking_status**: status de tabagismo (never smoked = nunca fumou, formerly smoked = ex-fumante, smokes = fumante, ou Unknown = desconhecido).

O conjunto de dados possui **5.110 observações** (pacientes) e 12 variáveis. A seguir, faremos a leitura do dataset e conduziremos uma análise estatística descritiva, explorando as variáveis. Em seguida, calcularemos a correlação entre idade e nível de glicose e ajustaremos um modelo de regressão linear simples para investigar a relação entre eles.

# Leitura dos dados

Vamos iniciar carregando o dataset CSV em um DataFrame pandas.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
```

```python
# Caminho do arquivo dentro do dataset
file_path = "healthcare-dataset-stroke-data.csv"

# Carregar o dataset diretamente do Kaggle como DataFrame pandas
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "fedesoriano/stroke-prediction-dataset",
    file_path,
)

# Visualizar as primeiras linhas
print("Dimensões do dataset:", df.shape)
df.head()
```

> Dimensões do dataset: (5110, 12)

| | id | gender | age | hypertension | heart_disease | ever_married | work_type | Residence_type | avg_glucose_level | bmi | smoking_status | stroke |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **0** | 9046 | Male | 67.0 | 0 | 1 | Yes | Private | Urban | 228.69 | 36.6 | formerly smoked | 1 |
| **1** | 51676 | Female | 61.0 | 0 | 0 | Yes | Self-employed | Rural | 202.21 | NaN | never smoked | 1 |
| **2** | 31112 | Male | 80.0 | 0 | 1 | Yes | Private | Rural | 105.92 | 32.5 | never smoked | 1 |
| **3** | 60182 | Female | 49.0 | 0 | 0 | Yes | Private | Urban | 171.23 | 34.4 | smokes | 1 |
| **4** | 1665 | Female | 79.0 | 1 | 0 | Yes | Self-employed | Rural | 174.12 | 24.0 | never smoked | 1 |

Como podemos ver, o dataset contém 5.110 registros e 12 colunas. Observe que valores ausentes são representados como N/A em algumas colunas (por exemplo, `bmi`). Vamos verificar quantos valores faltantes existem em cada variável:

```python
# Verificar valores ausentes em cada coluna
print(df.isna().sum())
```

```
id                     0
gender                 0
age                    0
hypertension           0
heart_disease          0
ever_married           0
work_type              0
Residence_type         0
avg_glucose_level      0
bmi                  201
smoking_status         0
stroke                 0
dtype: int64
```

No resultado acima, notamos que a variável `bmi` possui 201 valores ausentes. Para fins da análise descritiva, iremos ignorar esses valores (os métodos estatísticos do pandas já fazem essa exclusão automaticamente).

# Análise Descritiva – Variáveis Qualitativas e Quantitativas Discretas

Começamos explorando as variáveis categóricas e as variáveis quantitativas discretas (indicadores 0/1). As principais são:
* Gênero (`gender`),
* Já foi casado (`ever_married`),
* Tipo de trabalho (`work_type`),
* Tipo de residência (`Residence_type`),
* Status de tabagismo (`smoking_status`),
* Hipertensão (`hypertension`, 0/1),
* Doença cardíaca (`heart_disease`, 0/1),
* AVC (`stroke`, 0/1, variável de desfecho).

### Distribuições de frequência

```python
# Tabelas de frequência para variáveis categóricas e discretas
print("Distribuição por gênero:\n", df['gender'].value_counts(), "\n")
print("Estado civil (ever_married):\n", df['ever_married'].value_counts(), "\n")
print("Tipo de residência:\n", df['Residence_type'].value_counts(), "\n")
print("Hipertensão:\n", df['hypertension'].value_counts(), "\n")
print("Doença cardíaca:\n", df['heart_disease'].value_counts(), "\n")
print("AVC (0=Não, 1=Sim):\n", df['stroke'].value_counts(), "\n")
```

**Tabela 1: Distribuição de frequências** – Observando as saídas, podemos destacar:
* **Gênero**: A maioria dos pacientes são do sexo feminino (58,6%).
* **Estado civil**: Aproximadamente 65,6% dos pacientes já foram casados.
* **Tipo de residência**: A população está quase dividida entre áreas urbanas (50,8%) e rurais (49,2%).
* **Hipertensão**: Cerca de 9,7% dos pacientes têm histórico de hipertensão.
* **Doença cardíaca**: Apenas 5,4% apresentam doença cardíaca.
* **AVC**: Somente 249 pacientes (4,9%) tiveram AVC. Este dado evidencia que o desfecho é relativamente raro no dataset.

### Visualização de variáveis categóricas

```python
plt.figure(figsize=(8,5))
df['work_type'].value_counts().plot(kind='bar', color='cornflowerblue', edgecolor='black')
plt.title('Distribuição dos Pacientes por Tipo de Trabalho')
plt.xlabel('Tipo de Trabalho')
plt.ylabel('Frequência')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

![png](Trabalho_Final_PE_files/Trabalho_Final_PE_12_0.png)

*Figura 1: Distribuição dos pacientes por tipo de trabalho.*

Observa-se que a maioria (57,2%) trabalha no setor privado. Outro aspecto relevante é a proporção de pacientes que efetivamente sofreram AVC.

```python
plt.figure(figsize=(6,5))
df['stroke'].value_counts().sort_index().plot(kind='bar', color=['mediumseagreen', 'salmon'], edgecolor='black')
plt.title('Número de Pacientes com e sem AVC')
plt.xlabel('AVC (0 = Não, 1 = Sim)')
plt.ylabel('Quantidade')
plt.xticks([0,1], ['Não', 'Sim'], rotation=0)
plt.tight_layout()
plt.show()
```

![png](Trabalho_Final_PE_files/Trabalho_Final_PE_15_0.png)

*Figura 2: Comparação do número de pacientes que tiveram AVC (sim) versus que não tiveram (não).*

A disparidade é evidente, com o grupo "Não" muito mais numeroso.

# Análise Descritiva – Variáveis Quantitativas Contínuas

Passamos agora às variáveis quantitativas contínuas: idade (`age`), nível médio de glicose (`avg_glucose_level`) e IMC (`bmi`).

```python
# Medidas descritivas das variáveis contínuas
print(df[['age','avg_glucose_level','bmi']].describe())
```

**Tabela 2: Medidas descritivas (idade, glicose média, IMC)**

| Variável | N | Média | Desvio Padrão | Mínimo | Q1 | Mediana | Q3 | Máximo |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Idade (anos)** | 5110 | 43,23 | 22,61 | 0,08 | 25,0 | 45,0 | 61,0 | 82,0 |
| **Glicose (mg/dL)** | 5110 | 106,15 | 45,28 | 55,12 | 77,25 | 91,89 | 114,09 | 271,74 |
| **IMC (kg/m²)** | 4909 | 28,89 | 7,85 | 10,30 | 23,50 | 28,10 | 33,10 | 97,60 |

*Observação: O IMC tem 4909 casos válidos pois 201 estão ausentes (N/A).*

### Histogramas e Boxplots

```python
plt.figure(figsize=(8,5))
plt.hist(df['age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Distribuição da Idade dos Pacientes')
plt.xlabel('Idade (anos)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()
```

![png](Trabalho_Final_PE_files/Trabalho_Final_PE_22_0.png)

*Figura 3: Histograma da idade dos pacientes.*

```python
plt.figure(figsize=(8,5))
plt.hist(df['avg_glucose_level'], bins=30, color='salmon', edgecolor='black')
plt.title('Distribuição do Nível Médio de Glicose')
plt.xlabel('Glicose média (mg/dL)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()
```

![png](Trabalho_Final_PE_files/Trabalho_Final_PE_28_0.png)

*Figura 4: Histograma do nível médio de glicose. A distribuição é assimétrica à direita.*

```python
plt.figure(figsize=(8,5))
plt.hist(df['bmi'].dropna(), bins=30, color='lightgreen', edgecolor='black')
plt.title('Distribuição do IMC dos Pacientes')
plt.xlabel('IMC (kg/m²)')
plt.ylabel('Frequência')
plt.tight_layout()
plt.show()
```

![png](Trabalho_Final_PE_files/Trabalho_Final_PE_31_0.png)

*Figura 5: Histograma do IMC dos pacientes.*

```python
plt.figure(figsize=(15,5))
# Boxplot da Idade
plt.subplot(1, 3, 1)
sns.boxplot(y=df['age'], color='skyblue')
plt.title('Idade')
# ... e assim por diante para os outros plots
plt.suptitle('Boxplots das variáveis contínuas', fontsize=14)
plt.tight_layout()
plt.show()
```

![png](Trabalho_Final_PE_files/Trabalho_Final_PE_33_0.png)

*Figura 6: Boxplots de Idade, Glicose Média e IMC.*

Os boxplots confirmam a presença de muitos *outliers* (valores atípicos) para Glicose e IMC, indicando que alguns pacientes possuem valores muito elevados nessas variáveis.

# Correlação entre Idade e Nível de Glicose

Investigamos se pacientes mais velhos tendem a apresentar glicose média mais elevada.

```python
import numpy as np
# Calcular a correlação de Pearson entre idade e nível médio de glicose
corr = df['age'].corr(df['avg_glucose_level'])
print(f"Correlação de Pearson (idade vs glicose média) = {corr:.3f}")
```
> Correlação de Pearson (idade vs glicose média) = 0.238

O valor de 0,238 indica uma correlação positiva fraca entre idade e nível de glicose. Embora estatisticamente significativa, na prática a idade explica apenas cerca de $0,238^2 \approx 5,7\%$ da variação nos níveis de glicose.

# Regressão Linear Simples (Glicose ~ Idade)

Modelamos a relação entre as variáveis usando um modelo linear simples $Y = \beta_0 + \beta_1 X$, onde Y é a glicose e X é a idade.

```python
from sklearn.linear_model import LinearRegression

X = df[['age']].values
y = df['avg_glucose_level'].values

model = LinearRegression()
model.fit(X, y)

beta0 = model.intercept_
beta1 = model.coef_[0]
r2 = model.score(X, y)

print(f"Intercepto (beta0) = {beta0:.2f}")
print(f"Inclinação (beta1) = {beta1:.3f}")
print(f"R² do modelo = {r2:.3f}")
```

> Intercepto (beta0) = 85.53
> Inclinação (beta1) = 0.477
> R² do modelo = 0.057

A equação de regressão ajustada é:
$Glicose\ Média\ Estimada = 85,53 + 0,477 \times Idade$.

Isso sugere que, em média, a cada ano a mais de idade, o nível de glicose aumenta em 0,477 mg/dL. O $R^2$ de 0,057 confirma que o modelo tem baixo poder explicativo, como esperado pela correlação fraca.

```python
plt.figure(figsize=(8,5))
sns.scatterplot(x='age', y='avg_glucose_level', data=df, alpha=0.5)
sns.regplot(x='age', y='avg_glucose_level', data=df, scatter=False, color='red', label='Regressão Linear')
plt.title('Idade vs Nível Médio de Glicose')
plt.xlabel('Idade (anos)')
plt.ylabel('Glicose média (mg/dL)')
plt.legend()
plt.tight_layout()
plt.show()
```

![png](Trabalho_Final_PE_files/Trabalho_Final_PE_41_0.png)

*Figura 7: Diagrama de dispersão com a linha de regressão, mostrando a leve tendência positiva.*

# Conclusões

* A amostra é majoritariamente composta por mulheres de meia-idade, com baixa incidência de AVC (~5%).
* Fatores de risco como hipertensão (9,7%), sobrepeso/obesidade (64%) e glicemia elevada (~39% com pré-diabetes ou diabetes) são prevalentes.
* Existe uma correlação positiva fraca, mas significativa, entre idade e glicemia.
* A análise confirma que o perfil de risco da amostra está alinhado com o conhecimento médico sobre AVC, destacando a importância do controle de peso, glicemia e pressão arterial.

Estes resultados preparam o terreno para estudos posteriores, como a criação de modelos preditivos para identificar pacientes de alto risco.
