import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# --- 1. Configuração da Página e Estilos ---
st.set_page_config(
    page_title="Dashboard de Risco de AVC",
    page_icon="🧠",
    layout="wide"
)

# --- 2. Carregamento e Tradução dos Dados ---
@st.cache_data
def carregar_dados():
    """
    Carrega, limpa, traduz e prepara os dados de AVC.
    A execução é armazenada em cache para performance.
    """
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df = df[df['gender'] != 'Other']
    df.dropna(subset=['bmi'], inplace=True)

    # --- Mapeamentos para Tradução ---
    gender_map = {"Female": "Feminino", "Male": "Masculino"}
    work_type_map = {
        "Private": "Setor Privado",
        "Self-employed": "Autônomo",
        "Govt_job": "Servidor Público",
        "children": "Criança",
        "Never_worked": "Nunca Trabalhou"
    }
    smoking_status_map = {
        "formerly smoked": "Ex-fumante",
        "never smoked": "Nunca fumou",
        "smokes": "Fumante",
        "Unknown": "Desconhecido"
    }
    residence_map = {"Urban": "Urbana", "Rural": "Rural"}
    stroke_map = {0: "Não", 1: "Sim"}

    # --- Criação das Colunas Traduzidas ---
    df['genero_pt'] = df['gender'].map(gender_map)
    df['tipo_trabalho_pt'] = df['work_type'].map(work_type_map)
    df['status_tabagismo_pt'] = df['smoking_status'].map(smoking_status_map)
    df['tipo_residencia_pt'] = df['Residence_type'].map(residence_map)
    df['avc_pt'] = df['stroke'].map(stroke_map)

    return df

df = carregar_dados()

# --- 3. Barra Lateral com Filtros (Sidebar) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2666/2666598.png", width=80)
    st.title("Filtros Interativos")
    st.markdown("Ajuste os filtros abaixo para explorar os dados e descobrir insights.")

    faixa_idade = st.slider(
        "Faixa de Idade:",
        min_value=int(df["age"].min()),
        max_value=int(df["age"].max()),
        value=(int(df["age"].min()), int(df["age"].max()))
    )
    genero = st.multiselect("Gênero:", options=df["genero_pt"].unique(), default=df["genero_pt"].unique())
    tipo_trabalho = st.multiselect("Tipo de Trabalho:", options=df["tipo_trabalho_pt"].unique(), default=df["tipo_trabalho_pt"].unique())
    status_tabagismo = st.multiselect("Status de Tabagismo:", options=df["status_tabagismo_pt"].unique(), default=df["status_tabagismo_pt"].unique())
    st.markdown("---")
    hipertensao = st.checkbox("Apenas pacientes com hipertensão", value=False)
    doenca_cardiaca = st.checkbox("Apenas pacientes com doença cardíaca", value=False)

# --- 4. Aplicação dos Filtros ---
df_filtrado = df[
    (df["age"].between(faixa_idade[0], faixa_idade[1])) &
    (df["genero_pt"].isin(genero)) &
    (df["tipo_trabalho_pt"].isin(tipo_trabalho)) &
    (df["status_tabagismo_pt"].isin(status_tabagismo))
]
if hipertensao:
    df_filtrado = df_filtrado[df_filtrado["hypertension"] == 1]
if doenca_cardiaca:
    df_filtrado = df_filtrado[df_filtrado["heart_disease"] == 1]

# --- 5. Lógica de Prevenção de Erros ---
if df_filtrado.empty:
    st.warning("Nenhum dado encontrado para a combinação de filtros selecionada. Por favor, ajuste os filtros.")
    st.stop()

# --- 6. Conteúdo Principal do Dashboard com Abas ---
st.title("🧠 Análise Interativa de Fatores de Risco de AVC")
st.markdown("Este dashboard apresenta uma análise detalhada dos fatores demográficos e de saúde relacionados ao Acidente Vascular Cerebral (AVC).")

tab1, tab2, tab3 = st.tabs(["**Visão Geral & Descritiva**", "**Relações entre Variáveis**", "**Dados Detalhados**"])

# --- Aba 1: Visão Geral e Análise Descritiva ---
with tab1:
    st.header("Métricas da População Filtrada")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nº de Pacientes", f"{df_filtrado.shape[0]:,}")
    col2.metric("Idade Média", f"{df_filtrado['age'].mean():.1f} anos")
    col3.metric("Glicose Média", f"{df_filtrado['avg_glucose_level'].mean():.1f} mg/dL")
    col4.metric("Pacientes com AVC", f"{df_filtrado['stroke'].sum()} ({df_filtrado['stroke'].mean():.1%})")

    st.markdown("---")
    st.header("Análise das Variáveis")
    col_cat, col_quant = st.columns(2)
    with col_cat:
        st.subheader("Distribuições Categóricas")
        fig_gen, ax_gen = plt.subplots(figsize=(6, 4)); sns.countplot(data=df_filtrado, x='genero_pt', ax=ax_gen, palette='pastel', order=df_filtrado['genero_pt'].value_counts().index); ax_gen.set_title("Distribuição por Gênero"); ax_gen.set_xlabel("Gênero"); ax_gen.set_ylabel("Contagem"); st.pyplot(fig_gen)
        fig_res, ax_res = plt.subplots(figsize=(6, 4)); df_filtrado['tipo_residencia_pt'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax_res, colors=sns.color_palette('pastel')); ax_res.set_ylabel(''); ax_res.set_title("Distribuição por Tipo de Residência"); st.pyplot(fig_res)
    with col_quant:
        st.subheader("Distribuições Numéricas")
        fig_age, ax_age = plt.subplots(figsize=(6, 4)); sns.histplot(df_filtrado['age'], kde=True, ax=ax_age, color='skyblue', bins=20); ax_age.set_title("Histograma da Idade"); ax_age.set_xlabel("Idade"); ax_age.set_ylabel("Frequência"); st.pyplot(fig_age)
        fig_bmi, ax_bmi = plt.subplots(figsize=(6, 4)); sns.boxplot(x=df_filtrado['bmi'], ax=ax_bmi, color='lightgreen'); ax_bmi.set_title("Boxplot do IMC"); ax_bmi.set_xlabel("Índice de Massa Corporal (IMC)"); st.pyplot(fig_bmi)

# --- Aba 2: Relações entre Variáveis ---
with tab2:
    st.header("Matriz de Correlação")
    st.markdown("O mapa de calor abaixo mostra a correlação de Pearson entre as variáveis numéricas. Valores próximos de 1 (vermelho) indicam uma forte correlação positiva, enquanto valores próximos de -1 (azul) indicam uma forte correlação negativa.")

    cols_corr = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease', 'stroke']
    matriz_corr = df_filtrado[cols_corr].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
    sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr, linewidths=.5)
    ax_corr.set_title("Mapa de Calor de Correlações")
    st.pyplot(fig_corr)

    st.markdown("---")
    st.header("Análise Bivariada Interativa")
    st.markdown("Selecione duas variáveis para visualizar a relação entre elas. Marque a caixa para adicionar a análise de regressão linear.")

    opcoes_numericas = ['age', 'avg_glucose_level', 'bmi']
    opcoes_categoricas_hue = ['Nenhuma', 'genero_pt', 'avc_pt', 'tipo_residencia_pt']

    col_sel1, col_sel2, col_sel3 = st.columns(3)
    with col_sel1:
        x_axis = st.selectbox("Selecione a variável do Eixo X:", options=opcoes_numericas, index=0)
    with col_sel2:
        y_axis = st.selectbox("Selecione a variável do Eixo Y:", options=opcoes_numericas, index=1)
    with col_sel3:
        # O hue não funciona com regplot, então desabilitamos se a regressão for escolhida
        # Ou simplesmente o removemos da lógica do regplot
        hue_axis = st.selectbox("Colorir por (opcional):", options=opcoes_categoricas_hue, index=0)
    
    # Checkbox para ativar a regressão <--- ADICIONADO DE VOLTA
    show_regression = st.checkbox("Mostrar linha de regressão e coeficientes")

    # Plotar o gráfico
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
    
    # Lógica para escolher o gráfico correto
    if show_regression:
        sns.regplot(data=df_filtrado, x=x_axis, y=y_axis, ax=ax_scatter,
                    scatter_kws={'alpha':0.4}, line_kws={"color": "red"})
        ax_scatter.set_title(f"Regressão Linear: {x_axis.capitalize()} vs. {y_axis.capitalize()}")
    else:
        hue_param = None if hue_axis == 'Nenhuma' else hue_axis
        sns.scatterplot(data=df_filtrado, x=x_axis, y=y_axis, hue=hue_param, ax=ax_scatter, alpha=0.6, palette="viridis")
        ax_scatter.set_title(f"Gráfico de Dispersão: {x_axis.capitalize()} vs. {y_axis.capitalize()}")
    
    ax_scatter.set_xlabel(x_axis.replace('_', ' ').capitalize())
    ax_scatter.set_ylabel(y_axis.replace('_', ' ').capitalize())
    st.pyplot(fig_scatter)

    # Calcular e exibir estatísticas
    if df_filtrado.shape[0] > 1:
        corr_scatter, p_scatter = pearsonr(df_filtrado[x_axis], df_filtrado[y_axis])
        st.metric(f"Correlação entre {x_axis} e {y_axis}", f"{corr_scatter:.3f}",
                  help=f"P-valor: {p_scatter:.3g}. Um p-valor baixo (geralmente < 0.05) indica que a correlação é estatisticamente significativa.")

        # Se o checkbox de regressão estiver marcado, mostrar os coeficientes <--- LÓGICA ADICIONADA
        if show_regression:
            X = df_filtrado[[x_axis]]
            y = df_filtrado[y_axis]
            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)
            beta1 = model.coef_[0]
            beta0 = model.intercept_

            st.subheader("Resultados da Regressão Linear Simples")
            reg_col1, reg_col2, reg_col3 = st.columns(3)
            reg_col1.metric("Coeficiente Angular (β₁)", f"{beta1:.3f}")
            reg_col2.metric("Intercepto (β₀)", f"{beta0:.2f}")
            reg_col3.metric("Coeficiente de Determinação (R²)", f"{r2:.3f}")

# --- Aba 3: Dados Detalhados ---
with tab3:
    st.header("Tabela de Dados Filtrados")
    st.markdown("Visualize e explore os dados brutos correspondentes aos filtros aplicados.")
    st.dataframe(df_filtrado, use_container_width=True, hide_index=True)
