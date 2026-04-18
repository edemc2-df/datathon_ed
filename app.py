import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Risco de Defasagem - Passos Mágicos",
    page_icon="📘",
    layout="centered"
)

@st.cache_resource
def carregar_modelo():
    return joblib.load("model.pkl")

modelo = carregar_modelo()

FEATURES = ['IDA', 'IEG', 'IPS', 'IPP', 'IAN', 'IPV']

st.title("📘 Preditor de Risco de Defasagem")
st.subheader("Associação Passos Mágicos")
st.write(
    "Esta aplicação estima a probabilidade de um aluno estar em situação de risco "
    "com base em indicadores acadêmicos, de engajamento e desenvolvimento."
)

st.markdown("""
### 🎯 Objetivo
Apoiar a identificação precoce de alunos em risco, permitindo ações preventivas
pela equipe pedagógica com base em dados.
""")

st.markdown("---")
st.markdown("### Informe os indicadores do aluno")

col1, col2 = st.columns(2)

with col1:
    ida = st.number_input("IDA", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
    ieg = st.number_input("IEG", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
    ips = st.number_input("IPS", min_value=0.0, max_value=10.0, value=6.0, step=0.1)

with col2:
    ipp = st.number_input("IPP", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
    ian = st.number_input("IAN", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
    ipv = st.number_input("IPV", min_value=0.0, max_value=10.0, value=6.0, step=0.1)

st.markdown("---")

if st.button("Prever risco", use_container_width=True):
    entrada = pd.DataFrame([[ida, ieg, ips, ipp, ian, ipv]], columns=FEATURES)

    predicao = modelo.predict(entrada)[0]
    probabilidade = modelo.predict_proba(entrada)[0][1]

    st.markdown("### 📊 Resultado da análise")

    if probabilidade >= 0.70:
        st.error(f"🔴 Alto risco de defasagem ({probabilidade:.1%})")
        faixa = "Alto risco"
        interpretacao = "O aluno apresenta forte probabilidade de queda de desempenho."
    elif probabilidade >= 0.40:
        st.warning(f"🟡 Médio risco de defasagem ({probabilidade:.1%})")
        faixa = "Médio risco"
        interpretacao = "O aluno apresenta sinais de atenção e pode precisar de acompanhamento."
    else:
        st.success(f"🟢 Baixo risco de defasagem ({probabilidade:.1%})")
        faixa = "Baixo risco"
        interpretacao = "O aluno apresenta indicadores mais estáveis no cenário atual."

    st.write(f"**Classificação prevista pelo modelo:** {int(predicao)}")
    st.write(f"**Faixa interpretativa:** {faixa}")
    st.write(f"**Interpretação:** {interpretacao}")

    st.markdown("### 🔎 Fatores que influenciaram o resultado")

    fatores = []

    if ida < 6:
        fatores.append("Desempenho acadêmico baixo (IDA).")
    if ieg < 6:
        fatores.append("Baixo engajamento nas atividades (IEG).")
    if ipv < 6:
        fatores.append("Baixo indicador de ponto de virada (IPV).")
    if ips < 5:
        fatores.append("Aspectos psicossociais merecem atenção (IPS).")
    if ipp < 5:
        fatores.append("Aspectos psicopedagógicos merecem acompanhamento (IPP).")
    if ian < 6:
        fatores.append("Indicador de adequação de nível abaixo do ideal (IAN).")

    if fatores:
        for fator in fatores:
            st.write(f"• {fator}")
    else:
        st.write("Indicadores dentro de faixa adequada para o perfil analisado.")

    st.markdown("### Variáveis mais relevantes no modelo")
    importancias = pd.Series(
        modelo.feature_importances_,
        index=FEATURES
    ).sort_values(ascending=False)

    st.dataframe(
        importancias.reset_index().rename(
            columns={"index": "Indicador", 0: "Importância"}
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### 📈 Importância dos indicadores")
    fig, ax = plt.subplots(figsize=(8, 4))
    importancias.sort_values().plot(kind='barh', ax=ax)
    ax.set_xlabel("Importância")
    ax.set_ylabel("Indicador")
    ax.set_title("Peso relativo das variáveis no modelo")
    st.pyplot(fig)

    st.markdown("### Leitura gerencial")
    st.write(
        "O modelo considera principalmente desempenho acadêmico (IDA), "
        "engajamento (IEG) e ponto de virada (IPV) como fatores centrais "
        "para estimar o risco. Assim, a identificação precoce desses sinais "
        "pode apoiar intervenções mais assertivas da equipe pedagógica."
    )

st.markdown("---")
st.markdown("""
💡 *Este modelo auxilia na tomada de decisão educacional, permitindo intervenções
mais assertivas com base em dados.*
""")
st.caption("Aplicação desenvolvida para o Datathon - Passos Mágicos")