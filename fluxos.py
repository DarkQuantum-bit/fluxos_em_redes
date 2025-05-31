import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value

# === Configurações Iniciais ===
st.set_page_config(page_title="MS529 - Otimização de Fluxo de Caixa", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>MS529 - Otimização de Fluxo de Caixa</h1>
    <p style='text-align: center; font-size:18px;'>Simule cenários financeiros, visualize fluxos para tomar melhores decisões!</p>
""", unsafe_allow_html=True)

COR_PRINCIPAL = "#4B8BBE"

# === Parâmetros ===
setores = ['A', 'B', 'C', 'D', 'E', 'F']
periodos = [1, 2, 3]
st.sidebar.header("⚙️ Parâmetros de Simulação")
seed = st.sidebar.number_input("🔹 Seed Aleatória", min_value=0, value=42)
np.random.seed(seed)

st.sidebar.subheader("🔸 Demandas")
demanda_min = st.sidebar.number_input("Demanda Mínima (R$)", value=10000)
demanda_max = st.sidebar.number_input("Demanda Máxima (R$)", value=100000)

demandas = {(t, s): (np.random.randint(demanda_min, demanda_max) if s != 'A' else -np.random.randint(demanda_min, demanda_max))
            for t in periodos for s in setores}

st.sidebar.subheader("🔸 Parâmetros dos Fluxos")
cap_min = st.sidebar.number_input("Capacidade Mínima (R$)", value=30000)
cap_max = st.sidebar.number_input("Capacidade Máxima (R$)", value=120000)
custo_min = st.sidebar.number_input("Custo Unitário Mínimo", value=1.0)
custo_max = st.sidebar.number_input("Custo Unitário Máximo", value=3.0)
juros_min = st.sidebar.number_input("Juros Mínimo (%)", value=1.0)
juros_max = st.sidebar.number_input("Juros Máximo (%)", value=5.0)
M = st.sidebar.number_input("Penalização por Erro (M)", value=10.0)

fluxos = []
for i in setores:
    for j in setores:
        if i != j:
            fluxo = (i, j, np.random.randint(cap_min, cap_max),
                     np.round(np.random.uniform(custo_min, custo_max), 2),
                     np.round(np.random.uniform(juros_min/100, juros_max/100), 4))
            fluxos.append(fluxo)

st.markdown("---")
if st.button("📊 Ver Dados Aleatórios Gerados"):
    df_demandas = pd.DataFrame([
        {'Período': t, 'Setor': s, 'Demanda': demandas[(t, s)]} 
        for (t, s) in demandas
    ])
    st.dataframe(df_demandas)

    df_fluxos_aleatorios = pd.DataFrame(fluxos, columns=["De", "Para", "Capacidade", "Custo", "Juros"])
    st.dataframe(df_fluxos_aleatorios)

st.markdown("---")

if st.button("🚀 Resolver Otimização"):
    with st.spinner("⏳ Resolvendo o problema..."):
        prob = LpProblem("Fluxo_Caixa_Setorial", LpMinimize)
        x = LpVariable.dicts("x", ((i, j, t) for (i, j, _, _, _) in fluxos for t in periodos), lowBound=0)
        saldo = LpVariable.dicts("saldo", ((s, t) for s in setores for t in periodos), lowBound=0)
        erro = LpVariable.dicts("erro", ((s, t) for s in setores for t in periodos), lowBound=0)

        custo_total = []
        for (i, j, cap, custo, juros) in fluxos:
            for t in periodos:
                custo_fluxo = custo * x[i, j, t]
                custo_juros = juros * x[i, j, t]
                custo_total.append(custo_fluxo + custo_juros)

        penalidade_erro = lpSum(M * erro[s, t] for s in setores for t in periodos)
        prob += lpSum(custo_total) + penalidade_erro

        for (i, j, cap, _, _) in fluxos:
            for t in periodos:
                prob += x[i, j, t] <= cap

        for s in setores:
            for t in periodos:
                entradas = lpSum(x[i, s, t] for (i, j, _, _, _) in fluxos if j == s)
                saidas = lpSum(x[s, j, t] for (i, j, _, _, _) in fluxos if i == s)
                if t == 1:
                    saldo_prev = 0
                else:
                    saldo_prev = saldo[s, t-1]
                prob += entradas - saidas + saldo_prev + erro[s, t] == demandas.get((t, s), 0) + saldo[s, t]

        prob.solve()

    st.success(f"Status: {LpStatus[prob.status]} | Custo Total: R$ {value(prob.objective):,.2f}")

    fluxos_resultado = []
    erros_resultado = []
    for v in prob.variables():
        if "x_" in v.name and v.varValue > 0:
            partes = v.name.split("_")
            de = partes[1].strip("(),' ")
            para = partes[2].strip("(),' ")
            t = int(partes[3].strip("(),' "))
            fluxos_resultado.append([de, para, t, v.varValue])
        if "erro_" in v.name and v.varValue > 0:
            partes = v.name.split("_")
            setor = partes[1].strip("(),' ")
            t = int(partes[2].strip("(),' "))
            erros_resultado.append([setor, t, v.varValue])

    df_fluxos = pd.DataFrame(fluxos_resultado, columns=["De", "Para", "Período", "Fluxo"])
    df_erros = pd.DataFrame(erros_resultado, columns=["Setor", "Período", "Erro"])

    st.dataframe(df_fluxos)
    st.dataframe(df_erros)

    st.markdown(f"""<h3 style='color:{COR_PRINCIPAL};'>🌐 Grafo de Fluxos Encontrados</h3>""", unsafe_allow_html=True)
    G = nx.DiGraph()
    for s in setores:
        G.add_node(s)

    for _, row in df_fluxos.iterrows():
        i, j, t, f = row
        label = f"R${int(f):,} (t{t})"
        G.add_edge(i, j, label=label, weight=f)

    pos = nx.spring_layout(G, seed=42)
    edge_labels = {(i, j): G[i][j]['label'] for i, j in G.edges()}
    fig, ax = plt.subplots(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color="#4B8BBE", node_size=1500, font_color="white", font_weight="bold", edge_color="#ccc", arrowsize=20)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black", font_size=9)
    st.pyplot(fig)

    st.markdown(f"""<h3 style='color:{COR_PRINCIPAL};'>📝 Narrativa do Fluxo de Caixa</h3>""", unsafe_allow_html=True)
    texto_narrativo = """
    O setor A atua como fonte inicial de recursos, distribuindo fluxos para os demais setores (B a F) de acordo com as demandas e restrições de capacidade. A cada período, os saldos remanescentes são transportados para o próximo período, permitindo uma gestão eficiente do caixa. As demandas de cada setor foram atendidas parcialmente ou com erros, conforme necessário, minimizando o custo total. As penalidades aplicadas refletem a priorização de fluxos essenciais, enquanto os custos e juros foram considerados para reduzir os impactos financeiros. Este modelo permite uma compreensão visual e quantitativa das transferências financeiras entre os setores e períodos, apoiando decisões estratégicas baseadas em otimização matemática.
    """
    st.markdown(texto_narrativo)
