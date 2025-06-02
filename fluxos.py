import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import math
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value

# === Configurações Iniciais ===
st.set_page_config(page_title="MS529 - Otimização de Fluxo de Caixa", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>MS529 - Otimização de Fluxo de Caixa</h1>
    <p style='text-align: center; font-size:18px;'>Simule cenários financeiros, visualize fluxos para tomar melhores decisões!</p>
""", unsafe_allow_html=True)

COR_PRINCIPAL = "#007adb"

setores = ['A', 'B', 'C', 'D', 'E', 'F']
periodos = [1, 2, 3]

if 'modo_dados' not in st.session_state:
    st.session_state.modo_dados = "Gerar aleatoriamente"

if st.sidebar.button("🔄 Alternar modo de dados"):
    st.session_state.modo_dados = "Inserir manualmente" if st.session_state.modo_dados == "Gerar aleatoriamente" else "Gerar aleatoriamente"

modo_dados = st.session_state.modo_dados
st.sidebar.write(f"📝 Modo atual: **{modo_dados}**")

demandas = {}
fluxos = []

if modo_dados == "Gerar aleatoriamente":
    seed = st.sidebar.number_input("🔹 Seed aleatória", min_value=0, value=42)
    np.random.seed(seed)

    st.sidebar.subheader("🔸 Demandas aleatórias")
    demanda_total = st.sidebar.number_input("Demanda total por período (R$)", value=400000)

    st.sidebar.subheader("🔸 Parâmetros dos fluxos aleatórios")
    cap_min = st.sidebar.number_input("Capacidade mínima (R$)", value=30000)
    cap_max = st.sidebar.number_input("Capacidade máxima (R$)", value=120000)
    custo_min = st.sidebar.number_input("Custo unitário mínimo", value=1.0)
    custo_max = st.sidebar.number_input("Custo unitário máximo", value=3.0)
    juros_min = st.sidebar.number_input("Juros mínimo (%)", value=1.0)
    juros_max = st.sidebar.number_input("Juros máximo (%)", value=5.0)

    for t in periodos:
        proporcoes = np.random.dirichlet(np.ones(len(setores) - 1), 1).flatten()
        for idx, s in enumerate([x for x in setores if x != 'A']):
            demandas[(t, s)] = int(demanda_total * proporcoes[idx])
        demandas[(t, 'A')] = -sum(demandas[(t, s)] for s in setores if s != 'A')

    for i in setores:
        for j in setores:
            if i != j:
                cap = np.random.randint(cap_min, cap_max)
                custo = np.round(np.random.uniform(custo_min, custo_max), 2)
                juros = np.round(np.random.uniform(juros_min / 100, juros_max / 100), 4)
                fluxos.append((i, j, cap, custo, juros))
else:
    st.markdown("### 📊 Inserir demandas por período e setor")
    for t in periodos:
        st.markdown(f"#### Período {t}")
        cols = st.columns(len(setores))
        for idx, s in enumerate(setores):
            with cols[idx]:
                demandas[(t, s)] = st.number_input(f"Setor {s} (P{t})", value=0, step=1000, format="%d")

    st.markdown("### 📦 Inserir fluxos permitidos (Arestas)")
    for i in setores:
        for j in setores:
            if i != j:
                with st.expander(f"Fluxo de {i} para {j}"):
                    cols = st.columns(3)
                    with cols[0]:
                        cap = st.number_input(f"Capacidade {i}->{j}", value=50000, step=1000)
                    with cols[1]:
                        custo = st.number_input(f"Custo {i}->{j}", value=2.0, step=0.1, format="%.2f")
                    with cols[2]:
                        juros = st.number_input(f"Juros (%) {i}->{j}", value=3.0, step=0.1) / 100
                    fluxos.append((i, j, cap, custo, juros))

st.markdown("---")

st.markdown("### 📊 Demandas por período e setor")
st.dataframe(pd.DataFrame([
    {'Período': t, 'Setor': s, 'Demanda': demandas[(t, s)]} 
    for (t, s) in demandas
]))

st.markdown("### 📦 Fluxos permitidos (Arestas)")
st.dataframe(pd.DataFrame(fluxos, columns=["De", "Para", "Capacidade", "Custo", "Juros"]))

st.markdown("---")

if st.button("🚀 Otimizar"):
    M = st.sidebar.number_input("Penalização por erro (M)", value=10.0)
    for modo in ["Sem relaxamento", "Com relaxamento"]:
        st.markdown(f"## 🔍 {modo}")
        with st.spinner(f"⏳ Resolvendo o problema ({modo})..."):
            prob = LpProblem(f"Fluxo_Caixa_{modo}", LpMinimize)
            x = LpVariable.dicts("x", ((i, j, t) for (i, j, _, _, _) in fluxos for t in periodos), lowBound=0)
            saldo = LpVariable.dicts("saldo", ((s, t) for s in setores for t in periodos), lowBound=0)
            if modo == "Com relaxamento":
                erro = LpVariable.dicts("erro", ((s, t) for s in setores for t in periodos), lowBound=0)
            else:
                erro = {(s, t): 0 for s in setores for t in periodos}  # dummy

            custo_total = []
            for (i, j, cap, custo, juros) in fluxos:
                for t in periodos:
                    custo_fluxo = custo * x[i, j, t]
                    custo_juros = juros * x[i, j, t]
                    custo_total.append(custo_fluxo + custo_juros)
            if modo == "Com relaxamento":
                penalidade_erro = lpSum(M * erro[s, t] for s in setores for t in periodos)
                prob += lpSum(custo_total) + penalidade_erro
            else:
                prob += lpSum(custo_total)

            for (i, j, cap, _, _) in fluxos:
                for t in periodos:
                    prob += x[i, j, t] <= cap

            for s in setores:
                for t in periodos:
                    entradas = lpSum(x[i, s, t] for (i, j, _, _, _) in fluxos if j == s)
                    saidas = lpSum(x[s, j, t] for (i, j, _, _, _) in fluxos if i == s)
                    saldo_prev = 0 if t == 1 else saldo[s, t-1]
                    if modo == "Com relaxamento":
                        prob += entradas - saidas + saldo_prev + erro[s, t] == demandas.get((t, s), 0) + saldo[s, t]
                    else:
                        prob += entradas - saidas + saldo_prev == demandas.get((t, s), 0) + saldo[s, t]

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
        st.dataframe(df_fluxos)

        if erros_resultado:
            df_erros = pd.DataFrame(erros_resultado, columns=["Setor", "Período", "Erro"])
            st.markdown("### ⚠️ Demandas não atendidas")
            st.dataframe(df_erros)

        st.markdown(f"<h3 style='color:{COR_PRINCIPAL};'>🌐 Grafo ({modo})</h3>", unsafe_allow_html=True)
        G = nx.DiGraph()
        for s in setores:
            G.add_node(s)
        for (i, j, t, f) in df_fluxos.values:
            if G.has_edge(i, j):
                G[i][j]['label'] += f"\nR${int(f):,} (t{t})"
            else:
                G.add_edge(i, j, label=f"R${int(f):,} (t{t})", weight=f)

        pos = {'A': (0, 0)}
        for idx, s in enumerate(['B', 'C', 'D', 'E', 'F']):
            angle = 2 * math.pi * idx / 5
            pos[s] = (5 * np.cos(angle), 5 * np.sin(angle))

        edge_labels = {(i, j): G[i][j]['label'] for i, j in G.edges()}
        fig, ax = plt.subplots(figsize=(10, 8))
        nx.draw(G, pos, with_labels=True, node_color="#4B8BBE", node_size=1500, font_color="white", font_weight="bold", edge_color="#ccc", arrowsize=20)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black", font_size=9)
        st.pyplot(fig)
