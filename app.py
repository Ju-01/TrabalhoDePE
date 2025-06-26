import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import numpy as np

st.set_page_config(
    page_title="Dashboard Analítico de Risco de AVC",
    page_icon="🧠",
    layout="wide"
)

@st.cache_data
def carregar_dados():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    df = df[df['gender'] != 'Other']
    df.dropna(subset=['bmi'], inplace=True)

    gender_map = {"Female": "Feminino", "Male": "Masculino"}
    smoking_status_map = {"formerly smoked": "Ex-fumante", "never smoked": "Nunca fumou", "smokes": "Fumante", "Unknown": "Desconhecido"}
    stroke_map = {0: "Não", 1: "Sim"}
    hypertension_map = {0: "Não", 1: "Sim"}
    heart_disease_map = {0: "Não", 1: "Sim"}
    ever_married_map = {"Yes": "Já foi Casado(a)", "No": "Nunca foi Casado(a)"}

    df['genero_pt'] = df['gender'].map(gender_map)
    df['status_tabagismo_pt'] = df['smoking_status'].map(smoking_status_map)
    df['avc_pt'] = df['stroke'].map(stroke_map)
    df['hipertensao_pt'] = df['hypertension'].map(hypertension_map)
    df['doenca_cardiaca_pt'] = df['heart_disease'].map(heart_disease_map)
    df['casado_pt'] = df['ever_married'].map(ever_married_map)
    return df

df_original = carregar_dados()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2666/2666598.png", width=80)
    st.title("Controles de Análise")
    modo_comparativo = st.toggle("Ativar Modo de Análise Comparativa", help="Ative para comparar os perfis de quem teve e não teve AVC em cada gráfico.")
    st.markdown("---")
    st.title("Filtros de Dados")
    faixa_idade = st.slider("Faixa de Idade:", int(df_original["age"].min()), int(df_original["age"].max()), (int(df_original["age"].min()), int(df_original["age"].max())))
    genero = st.multiselect("Gênero:", df_original["genero_pt"].unique(), default=df_original["genero_pt"].unique())

df_filtrado = df_original[(df_original["age"].between(faixa_idade[0], faixa_idade[1])) & (df_original["genero_pt"].isin(genero))]

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado para a combinação de filtros selecionada.")
    st.stop()

st.title("🧠 Dashboard Analítico de Fatores de Risco de AVC")

tab1, tab2, tab3 = st.tabs(["**Análise de Perfis**", "**Relações e Regressão**", "**Dados Detalhados**"])

# --- Aba 1: Análise de Perfis (LAYOUT CORRIGIDO E ALINHADO) ---
with tab1:
    if modo_comparativo:
        st.header("Análise Comparativa de Perfis (AVC vs. Sem AVC)")
        if df_filtrado['stroke'].nunique() < 2:
            st.error("A seleção de filtros atual não contém pacientes com e sem AVC para comparação.")
            st.stop()

        st.subheader("Perfil Numérico (Distribuições)")
        col1, col2, col3 = st.columns(3)
        with col1:
            fig, ax = plt.subplots(); sns.kdeplot(data=df_filtrado, x='age', hue='avc_pt', fill=True, common_norm=False, palette='viridis', ax=ax); ax.set_xlabel("Idade"); ax.set_title("Distribuição de Idade"); st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(); sns.boxplot(data=df_filtrado, x='avc_pt', y='avg_glucose_level', palette='viridis', ax=ax); ax.set_xlabel("Teve AVC?"); ax.set_title("Nível de Glicose"); st.pyplot(fig)
        with col3:
            fig, ax = plt.subplots(); sns.boxplot(data=df_filtrado, x='avc_pt', y='bmi', palette='viridis', ax=ax); ax.set_xlabel("Teve AVC?"); ax.set_title("Índice de Massa Corporal"); st.pyplot(fig)

        st.markdown("---")
        st.subheader("Perfil Categórico (Proporções)")

        def plot_pie_pair(column, title):
            st.markdown(f"**{title}**")
            pie_col1, pie_col2 = st.columns(2)
            # Grupo SEM AVC
            with pie_col1:
                df_group = df_filtrado[df_filtrado['stroke'] == 0]
                if not df_group.empty:
                    counts = df_group[column].value_counts()
                    fig, ax = plt.subplots(); ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel')); ax.set_title("Grupo: Sem AVC"); st.pyplot(fig)
            # Grupo COM AVC
            with pie_col2:
                df_group = df_filtrado[df_filtrado['stroke'] == 1]
                if not df_group.empty:
                    counts = df_group[column].value_counts()
                    fig, ax = plt.subplots(); ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel')); ax.set_title("Grupo: Com AVC"); st.pyplot(fig)

        plot_pie_pair('hipertensao_pt', 'Comparativo de Hipertensão')
        plot_pie_pair('doenca_cardiaca_pt', 'Comparativo de Doença Cardíaca')
        plot_pie_pair('casado_pt', 'Comparativo de Estado Civil')
        plot_pie_pair('status_tabagismo_pt', 'Comparativo de Status de Tabagismo')

    else:
        st.header("Perfil Geral da População Selecionada")
        def plot_pie_chart(data, column, title):
            st.subheader(title)
            counts = data[column].value_counts(); fig, ax = plt.subplots(); ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis', len(counts))); st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            plot_pie_chart(df_filtrado, 'genero_pt', 'Proporção de Gênero')
            plot_pie_chart(df_filtrado, 'hipertensao_pt', 'Proporção de Hipertensão')
        with col2:
            plot_pie_chart(df_filtrado, 'status_tabagismo_pt', 'Proporção de Status de Tabagismo')
            plot_pie_chart(df_filtrado, 'doenca_cardiaca_pt', 'Proporção de Doença Cardíaca')

# --- Aba 2: Relações e Regressão ---
with tab2:
    st.header("Análise de Correlações entre Fatores de Risco")
    modo_heatmap = st.selectbox("Filtrar Heatmap por:",["População Geral (filtrada)", "Apenas Pacientes com AVC", "Apenas Pacientes sem AVC"])
    df_heatmap = df_filtrado.copy()
    if modo_heatmap == "Apenas Pacientes com AVC": df_heatmap = df_heatmap[df_heatmap['stroke'] == 1]
    elif modo_heatmap == "Apenas Pacientes sem AVC": df_heatmap = df_heatmap[df_heatmap['stroke'] == 0]
    if df_heatmap.empty or df_heatmap.shape[0] < 2: st.warning("Não há dados suficientes para gerar o heatmap.")
    else:
        df_heatmap_num = df_heatmap.copy();
        for col in ['genero_pt', 'casado_pt']:
            if col in df_heatmap_num.columns: df_heatmap_num[col] = pd.factorize(df_heatmap_num[col])[0]
        cols_heatmap = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease', 'genero_pt', 'casado_pt']
        matriz_corr_full = df_heatmap_num[cols_heatmap].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8)); sns.heatmap(matriz_corr_full, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr, linewidths=.5); st.pyplot(fig_corr)
    st.markdown("---")
    st.header("Análise de Regressão Linear Interativa")
    opcoes_numericas = ['age', 'avg_glucose_level', 'bmi']; col_sel1, col_sel2 = st.columns(2)
    x_axis = col_sel1.selectbox("Eixo X:", options=opcoes_numericas, index=0); y_axis = col_sel2.selectbox("Eixo Y:", options=opcoes_numericas, index=2)
    modo_analise = st.radio("Modo de Análise:",["Dispersão Simples", "Regressão Comparativa (AVC vs. Sem AVC)", "Regressão com Foco: Apenas Pacientes com AVC"], horizontal=True)
    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6));
    if modo_analise == "Dispersão Simples": sns.scatterplot(data=df_filtrado, x=x_axis, y=y_axis, hue='avc_pt', ax=ax_scatter, alpha=0.6, palette="viridis")
    elif modo_analise == "Regressão Comparativa (AVC vs. Sem AVC)":
        if df_filtrado['stroke'].nunique() < 2: st.warning("Modo comparativo indisponível.")
        else:
            sns.regplot(data=df_filtrado[df_filtrado['stroke']==0], x=x_axis, y=y_axis, ax=ax_scatter, scatter_kws={'alpha': 0.2}, label="Sem AVC"); sns.regplot(data=df_filtrado[df_filtrado['stroke']==1], x=x_axis, y=y_axis, ax=ax_scatter, scatter_kws={'alpha': 0.6}, label="Com AVC"); ax_scatter.legend()
    elif modo_analise == "Regressão com Foco: Apenas Pacientes com AVC":
        df_foco = df_filtrado[df_filtrado['stroke'] == 1]
        if df_foco.empty: st.warning("Não há pacientes com AVC na seleção.")
        else: sns.regplot(data=df_foco, x=x_axis, y=y_axis, ax=ax_scatter, scatter_kws={'alpha': 0.6}, line_kws={'color':'red'})
    ax_scatter.set_title(f"Análise: {x_axis.capitalize()} vs. {y_axis.capitalize()}"); st.pyplot(fig_scatter)
    def calcular_stats_regressao(dataframe):
        if dataframe.shape[0] > 1:
            X = dataframe[[x_axis]]; y = dataframe[y_axis]; valid_indices = ~np.isnan(X).any(axis=1) & ~np.isnan(y); X = X[valid_indices]; y = y[valid_indices]
            if len(X) > 1:
                model = LinearRegression().fit(X, y); r2 = model.score(X, y); beta1, beta0 = model.coef_[0], model.intercept_; corr, p_val = pearsonr(X.iloc[:,0], y)
                return corr, beta1, r2
        return None, None, None
    if 'Regressão' in modo_analise:
        st.subheader("Resultados da Análise de Regressão")
        if modo_analise == "Regressão Comparativa (AVC vs. Sem AVC)" and df_filtrado['stroke'].nunique() >= 2:
            col_res1, col_res2 = st.columns(2)
            corr_nao, beta1_nao, r2_nao = calcular_stats_regressao(df_filtrado[df_filtrado['stroke'] == 0])
            with col_res1: st.markdown("##### Grupo: Sem AVC");
            if corr_nao is not None: st.metric(label=f"Correlação", value=f"{corr_nao:.3f}"); st.metric(label=f"Coeficiente Angular (β₁)", value=f"{beta1_nao:.3f}"); st.metric(label=f"R²", value=f"{r2_nao:.3f}")
            corr_sim, beta1_sim, r2_sim = calcular_stats_regressao(df_filtrado[df_filtrado['stroke'] == 1])
            with col_res2: st.markdown("##### Grupo: Com AVC");
            if corr_sim is not None: st.metric(label=f"Correlação", value=f"{corr_sim:.3f}"); st.metric(label=f"Coeficiente Angular (β₁)", value=f"{beta1_sim:.3f}"); st.metric(label=f"R²", value=f"{r2_sim:.3f}")
        elif modo_analise == "Regressão com Foco: Apenas Pacientes com AVC":
            df_foco = df_filtrado[df_filtrado['stroke'] == 1]
            if not df_foco.empty:
                corr, beta1, r2 = calcular_stats_regressao(df_foco)
                if corr is not None:
                    col_res1, col_res2, col_res3 = st.columns(3); col_res1.metric(label=f"Correlação", value=f"{corr:.3f}"); col_res2.metric(label=f"Coeficiente Angular (β₁)", value=f"{beta1:.3f}"); col_res3.metric(label=f"R²", value=f"{r2:.3f}")

# --- Aba 3: Dados Detalhados ---
with tab3:
    st.header("Tabela de Dados Filtrados")
    st.dataframe(df_filtrado, use_container_width=True, hide_index=True)
