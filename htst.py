import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Doença Cardíaca",
    page_icon="❤️",
    layout="wide"
)

# CSS enxuto: só arruma espaçamento e deixa os cards com cara de card
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label { color: #6b7280; }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1rem;
    }
    .resultado {
        padding: 16px 20px;
        border-radius: 10px;
        font-size: 1.05rem;
        font-weight: 600;
        margin-top: 12px;
    }
    .doente   { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
    .saudavel { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
    h1 { font-size: 1.8rem !important; }
    h3 { margin-top: 1.5rem !important; }
</style>
""", unsafe_allow_html=True)


# ---------- Dados ----------
@st.cache_data
def carregar_dados():
    df = pd.read_csv('heart.csv')
    df = df.rename(columns={'chol': 'coleste', 'fbs': 'acucars', 'cp': 'dorpeito'})
    return df

@st.cache_resource
def treinar(df):
    X = df.drop(columns='target')
    y = df['target']
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(random_state=42).fit(X_tr, y_tr)
    lr = LogisticRegression(penalty='l2', solver='liblinear').fit(X_tr, y_tr)

    acc_rf = accuracy_score(y_te, rf.predict(X_te))
    acc_lr = accuracy_score(y_te, lr.predict(X_te))
    return rf, lr, scaler, acc_rf, acc_lr


df = carregar_dados()
rf_model, lr_model, scaler, acc_rf, acc_lr = treinar(df)


def prever(dado, modelo):
    x = scaler.transform([dado])
    pred = modelo.predict(x)[0]
    prob = modelo.predict_proba(x)[0][1]
    return pred, prob


def mostrar_resultado(pred, prob, nome):
    if pred == 1:
        st.markdown(
            f'<div class="resultado doente">⚠️ {nome}: provável doença cardíaca '
            f'(probabilidade {prob*100:.0f}%)</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="resultado saudavel">✅ {nome}: sem indicativo de doença '
            f'(probabilidade {prob*100:.0f}%)</div>',
            unsafe_allow_html=True
        )


# ---------- Sidebar ----------
st.sidebar.title("Menu")
pagina = st.sidebar.radio("Ir para", ["Início", "Análises", "Predição"])

st.sidebar.markdown("---")
st.sidebar.caption(
    f"{len(df)} pacientes no dataset  \n"
    f"Random Forest: {acc_rf*100:.1f}% de acurácia  \n"
    f"Regressão Logística: {acc_lr*100:.1f}% de acurácia"
)


# ---------- Página: Início ----------
if pagina == "Início":
    st.title("Doença Cardíaca")
    st.write("Análise e predição de risco cardíaco a partir do dataset Heart Disease.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Pacientes", len(df))
    c2.metric("Com doença", int(df['target'].sum()))
    c3.metric("Sem doença", int((df['target'] == 0).sum()))

    st.subheader("O que tem aqui")
    st.write(
        "- **Análises**: distribuições, correlações e os dados brutos.\n"
        "- **Predição**: você preenche os dados do paciente e o modelo responde."
    )


# ---------- Página: Análises ----------
elif pagina == "Análises":
    st.title("Análises")

    st.subheader("Distribuição do target")
    contagem = df['target'].value_counts().rename({0: 'Sem doença', 1: 'Com doença'})
    st.bar_chart(contagem)

    st.subheader("Idade por condição")
    st.bar_chart(df.groupby('age')['target'].mean())

    st.subheader("Correlação entre variáveis")
    st.dataframe(df.corr().round(2), use_container_width=True)

    st.subheader("Dados brutos")
    st.dataframe(df.head(50), use_container_width=True)


# ---------- Página: Predição ----------
else:
    st.title("Predição")
    st.write("Preencha os dados do paciente e escolha o modelo.")

    with st.form("form_paciente"):
        st.subheader("Dados do paciente")
        c1, c2 = st.columns(2)

        with c1:
            idade = st.slider("Idade", 20, 80, 50)
            sexo = st.selectbox("Sexo", [0, 1], format_func=lambda x: "Feminino" if x == 0 else "Masculino")
            dorpeito = st.slider("Dor no peito (0-3)", 0, 3, 1)
            pressao = st.slider("Pressão arterial em repouso", 80, 200, 120)
            coleste = st.slider("Colesterol", 100, 400, 200)
            acucars = st.selectbox("Açúcar no sangue > 120", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
            ecg = st.slider("Eletrocardiograma (0-2)", 0, 2, 1)

        with c2:
            freq = st.slider("Freq. cardíaca máxima", 60, 220, 150)
            angina = st.selectbox("Angina por exercício", [0, 1], format_func=lambda x: "Não" if x == 0 else "Sim")
            depressao = st.slider("Depressão ST", 0.0, 6.0, 1.0, step=0.1)
            inclinacao = st.slider("Inclinação do segmento ST (0-2)", 0, 2, 1)
            vasos = st.slider("Nº de vasos principais (0-4)", 0, 4, 0)
            thal = st.slider("Thalassemia (1-3)", 1, 3, 2)

        modelo = st.radio("Modelo", ["Random Forest", "Regressão Logística"], horizontal=True)
        enviado = st.form_submit_button("Prever")

    if enviado:
        dado = [idade, sexo, dorpeito, pressao, coleste, acucars,
                ecg, freq, angina, depressao, inclinacao, vasos, thal]

        if modelo == "Random Forest":
            pred, prob = prever(dado, rf_model)
        else:
            pred, prob = prever(dado, lr_model)

        mostrar_resultado(pred, prob, modelo)

    st.caption("Isso não substitui avaliação médica.")