import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import numpy as np

# --- 1. Configuração da Página e Estilos ---
st.set_page_config(
    page_title="Dashboard Comparativo de Risco de AVC",
    page_icon="🧠",
    layout="wide"
)

# --- 2. Carregamento e Tradução dos Dados ---
@st.cache_data
def carregar_dados():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df = df[df['gender'] != 'Other']
    df.dropna(subset=['bmi'], inplace=True)
    
    # Mapeamentos para Tradução (incluindo 'ever_married')
    gender_map = {"Female": "Feminino", "Male": "Masculino"}
    work_type_map = {"Private": "Setor Privado", "Self-employed": "Autônomo", "Govt_job": "Servidor Público", "children": "Criança", "Never_worked": "Nunca Trabalhou"}
    smoking_status_map = {"formerly smoked": "Ex-fumante", "never smoked": "Nunca fumou", "smokes": "Fumante", "Unknown": "Desconhecido"}
    residence_map = {"Urban": "Urbana", "Rural": "Rural"}
    stroke_map = {0: "Não", 1: "Sim"}
    hypertension_map = {0: "Não", 1: "Sim"}
    heart_disease_map = {0: "Não", 1: "Sim"}
    ever_married_map = {"Yes": "Já foi Casado(a)", "No": "Nunca foi Casado(a)"}

    # Criação das Colunas Traduzidas
    df['genero_pt'] = df['gender'].map(gender_map)
    df['tipo_trabalho_pt'] = df['work_type'].map(work_type_map)
    df['status_tabagismo_pt'] = df['smoking_status'].map(smoking_status_map)
    df['tipo_residencia_pt'] = df['Residence_type'].map(residence_map)
    df['avc_pt'] = df['stroke'].map(stroke_map)
    df['hipertensao_pt'] = df['hypertension'].map(hypertension_map)
    df['doenca_cardiaca_pt'] = df['heart_disease'].map(heart_disease_map)
    df['casado_pt'] = df['ever_married'].map(ever_married_map)
    return df

df = carregar_dados()

# --- 3. Barra Lateral com Filtros (Sidebar) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2666/2666598.png", width=80)
    st.title("Filtros Interativos")
    st.markdown("Ajuste os filtros abaixo para explorar os dados.")
    faixa_idade = st.slider("Faixa de Idade:", int(df["age"].min()), int(df["age"].max()), (int(df["age"].min()), int(df["age"].max())))
    genero = st.multiselect("Gênero:", df["genero_pt"].unique(), default=df["genero_pt"].unique())
    tipo_trabalho = st.multiselect("Tipo de Trabalho:", df["tipo_trabalho_pt"].unique(), default=df["tipo_trabalho_pt"].unique())
    status_tabagismo = st.multiselect("Status de Tabagismo:", df["status_tabagismo_pt"].unique(), default=df["status_tabagismo_pt"].unique())
    st.markdown("---")
    hipertensao = st.checkbox("Apenas com hipertensão", value=False)
    doenca_cardiaca = st.checkbox("Apenas com doença cardíaca", value=False)

# --- 4. Aplicação dos Filtros ---
df_filtrado = df[(df["age"].between(faixa_idade[0], faixa_idade[1])) & (df["genero_pt"].isin(genero)) & (df["tipo_trabalho_pt"].isin(tipo_trabalho)) & (df["status_tabagismo_pt"].isin(status_tabagismo))]
if hipertensao: df_filtrado = df_filtrado[df_filtrado["hypertension"] == 1]
if doenca_cardiaca: df_filtrado = df_filtrado[df_filtrado["heart_disease"] == 1]

# --- 5. Lógica de Prevenção de Erros ---
if df_filtrado.empty:
    st.warning("Nenhum dado encontrado para a combinação de filtros selecionada.")
    st.stop()

# --- 6. Conteúdo Principal do Dashboard com Abas ---
st.title("🧠 Análise Comparativa dos Fatores de Risco de AVC")
st.markdown("Este dashboard compara o perfil de pacientes que **tiveram AVC** com aqueles que **não tiveram**.")

tab1, tab2, tab3 = st.tabs(["**Análise Comparativa de Perfis**", "**Relações e Regressão**", "**Dados Detalhados**"])

# --- Aba 1: Análise Comparativa de Perfis (VERSÃO COM MAIS GRÁFICOS) ---
with tab1:
    st.header("Comparação do Perfil Demográfico e de Saúde")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Distribuição de Idade")
        fig, ax = plt.subplots(); sns.kdeplot(data=df_filtrado, x='age', hue='avc_pt', fill=True, common_norm=False, palette='viridis', ax=ax); ax.set_xlabel("Idade"); ax.set_ylabel("Densidade"); ax.legend(title='Teve AVC?', labels=['Sim', 'Não']); st.pyplot(fig)
        
        st.subheader("Proporção de Gênero")
        crosstab_gen = pd.crosstab(df_filtrado['genero_pt'], df_filtrado['avc_pt'], normalize='index')
        fig, ax = plt.subplots(); crosstab_gen.plot(kind='barh', stacked=True, color=sns.color_palette('viridis', 2), ax=ax, figsize=(8, 4)); ax.set_xlabel("Proporção"); ax.set_ylabel("Gênero"); ax.legend(title='Teve AVC?'); ax.set_xlim(0, 1); st.pyplot(fig)

    with col2:
        st.subheader("Nível de Glicose")
        fig, ax = plt.subplots(); sns.boxplot(data=df_filtrado, x='avc_pt', y='avg_glucose_level', palette='viridis', ax=ax); ax.set_xlabel("Teve AVC?"); ax.set_ylabel("Nível Médio de Glicose"); st.pyplot(fig)

        st.subheader("Proporção de Hipertensão")
        crosstab_hyper = pd.crosstab(df_filtrado['hipertensao_pt'], df_filtrado['avc_pt'], normalize='index')
        fig, ax = plt.subplots(); crosstab_hyper.plot(kind='barh', stacked=True, color=sns.color_palette('viridis', 2), ax=ax, figsize=(8, 4)); ax.set_xlabel("Proporção"); ax.set_ylabel("Tem Hipertensão?"); ax.legend(title='Teve AVC?'); ax.set_xlim(0, 1); st.pyplot(fig)

    with col3:
        st.subheader("Índice de Massa Corporal (IMC)")
        fig, ax = plt.subplots(); sns.boxplot(data=df_filtrado, x='avc_pt', y='bmi', palette='viridis', ax=ax); ax.set_xlabel("Teve AVC?"); ax.set_ylabel("IMC"); st.pyplot(fig)

        st.subheader("Proporção de Doença Cardíaca")
        crosstab_heart = pd.crosstab(df_filtrado['doenca_cardiaca_pt'], df_filtrado['avc_pt'], normalize='index')
        fig, ax = plt.subplots(); crosstab_heart.plot(kind='barh', stacked=True, color=sns.color_palette('viridis', 2), ax=ax, figsize=(8, 4)); ax.set_xlabel("Proporção"); ax.set_ylabel("Tem Doença Cardíaca?"); ax.legend(title='Teve AVC?'); ax.set_xlim(0, 1); st.pyplot(fig)

    st.markdown("---")
    st.header("Análise de Estilo de Vida e Estado Civil")
    col_vida1, col_vida2 = st.columns(2)
    with col_vida1:
        st.subheader("Proporção de Status de Tabagismo")
        crosstab_smoke = pd.crosstab(df_filtrado['status_tabagismo_pt'], df_filtrado['avc_pt'], normalize='index')
        fig, ax = plt.subplots(); crosstab_smoke.plot(kind='barh', stacked=True, color=sns.color_palette('viridis', 2), ax=ax, figsize=(8, 6)); ax.set_xlabel("Proporção"); ax.set_ylabel("Status de Tabagismo"); ax.legend(title='Teve AVC?'); ax.set_xlim(0, 1); st.pyplot(fig)
    with col_vida2:
        st.subheader("Proporção de Estado Civil")
        crosstab_married = pd.crosstab(df_filtrado['casado_pt'], df_filtrado['avc_pt'], normalize='index')
        fig, ax = plt.subplots(); crosstab_married.plot(kind='barh', stacked=True, color=sns.color_palette('viridis', 2), ax=ax, figsize=(8, 6)); ax.set_xlabel("Proporção"); ax.set_ylabel("Estado Civil"); ax.legend(title='Teve AVC?'); ax.set_xlim(0, 1); st.pyplot(fig)

# --- Aba 2: Relações e Regressão (VERSÃO COM SELETOR DE MODO) ---
with tab2:
    st.header("Análise Bivariada Interativa")
    st.markdown("Selecione duas variáveis para visualizar a relação entre elas e escolha um modo de análise.")
    
    opcoes_numericas = ['age', 'avg_glucose_level', 'bmi']
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        x_axis = st.selectbox("Selecione a variável do Eixo X:", options=opcoes_numericas, index=0)
    with col_sel2:
        y_axis = st.selectbox("Selecione a variável do Eixo Y:", options=opcoes_numericas, index=2)
    
    # Seletor de modo para a análise
    modo_analise = st.radio(
        "Escolha o Modo de Análise:",
        ["Dispersão Simples", "Regressão Comparativa (AVC vs. Sem AVC)", "Regressão com Foco: Apenas Pacientes com AVC"],
        horizontal=True
    )
    
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))

    # Lógica baseada no modo de análise escolhido
    if modo_analise == "Dispersão Simples":
        hue_axis = st.selectbox("Colorir por (opcional):", options=['Nenhuma', 'genero_pt', 'avc_pt'], index=2)
        hue_param = None if hue_axis == 'Nenhuma' else hue_axis
        sns.scatterplot(data=df_filtrado, x=x_axis, y=y_axis, hue=hue_param, ax=ax_scatter, alpha=0.6, palette="viridis")
        ax_scatter.set_title(f"Gráfico de Dispersão: {x_axis.capitalize()} vs. {y_axis.capitalize()}")

    elif modo_analise == "Regressão Comparativa (AVC vs. Sem AVC)":
        cores = {"Sim": "#fde725", "Não": "#440154"}
        for group_name, group_df in df_filtrado.groupby('avc_pt'):
            sns.regplot(data=group_df, x=x_axis, y=y_axis, ax=ax_scatter, scatter_kws={'alpha': 0.4}, label=f"AVC: {group_name}", color=cores.get(group_name))
        ax_scatter.legend()
        ax_scatter.set_title(f"Regressão Comparativa: {x_axis.capitalize()} vs. {y_axis.capitalize()}")
    
    elif modo_analise == "Regressão com Foco: Apenas Pacientes com AVC":
        df_foco = df_filtrado[df_filtrado['stroke'] == 1]
        if df_foco.empty:
            st.warning("Não há pacientes com AVC na seleção de filtros atual para realizar esta análise.")
            st.stop()
        sns.regplot(data=df_foco, x=x_axis, y=y_axis, ax=ax_scatter, scatter_kws={'alpha': 0.6}, line_kws={"color": "red"})
        ax_scatter.set_title(f"Regressão (Foco em AVC): {x_axis.capitalize()} vs. {y_axis.capitalize()}")

    ax_scatter.set_xlabel(x_axis.replace('_', ' ').capitalize())
    ax_scatter.set_ylabel(y_axis.replace('_', ' ').capitalize())
    st.pyplot(fig_scatter)

    # Exibição dos resultados estatísticos da regressão
    if 'Regressão' in modo_analise:
        st.subheader("Resultados da Análise de Regressão")
        
        if modo_analise == "Regressão Comparativa (AVC vs. Sem AVC)":
            col_res1, col_res2 = st.columns(2)
            # Função para evitar repetição de código
            def calcular_e_mostrar_stats(dataframe, coluna):
                if dataframe.shape[0] > 1:
                    X = dataframe[[x_axis]]; y = dataframe[y_axis]; valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y); X = X[valid_indices]; y = y[valid_indices]
                    if X.empty or y.empty: return
                    model = LinearRegression().fit(X, y); r2 = model.score(X, y); beta1, beta0 = model.coef_[0], model.intercept_; corr, p_val = pearsonr(X[x_axis], y)
                    with coluna:
                        st.markdown(f"#### Grupo: AVC {dataframe['avc_pt'].iloc[0]}")
                        st.metric(f"Correlação", f"{corr:.3f}"); st.metric(f"Coeficiente Angular (β₁)", f"{beta1:.3f}"); st.metric(f"R²", f"{r2:.3f}")

            calcular_e_mostrar_stats(df_filtrado[df_filtrado['stroke'] == 0], col_res1)
            calcular_e_mostrar_stats(df_filtrado[df_filtrado['stroke'] == 1], col_res2)
        
        elif modo_analise == "Regressão com Foco: Apenas Pacientes com AVC":
            df_foco = df_filtrado[df_filtrado['stroke'] == 1]
            if df_foco.shape[0] > 1:
                X = df_foco[[x_axis]]; y = df_foco[y_axis]; valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y); X = X[valid_indices]; y = y[valid_indices]
                model = LinearRegression().fit(X, y); r2 = model.score(X, y); beta1, beta0 = model.coef_[0], model.intercept_; corr, p_val = pearsonr(X[x_axis], y)
                st.markdown("#### Grupo: Apenas Pacientes com AVC")
                res_col1, res_col2, res_col3 = st.columns(3)
                res_col1.metric(f"Correlação", f"{corr:.3f}"); res_col2.metric(f"Coeficiente Angular (β₁)", f"{beta1:.3f}"); res_col3.metric(f"R²", f"{r2:.3f}")


# --- Aba 3: Dados Detalhados ---
with tab3:
    st.header("Tabela de Dados Filtrados")
    st.dataframe(df_filtrado, use_container_width=True, hide_index=True)
