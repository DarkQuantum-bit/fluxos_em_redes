import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value

# === Configurações iniciais ===
st.set_page_config(page_title="Otimização de Fluxo de Caixa", layout="wide")
st.title("MS529 - Otimização de Fluxo de Caixa com Fluxos em Redes")

# Setores e períodos fixos
setores = ['A', 'B', 'C', 'D', 'E', 'F']
periodos = [1, 2, 3]

# === Gerar Database Aleatória ===
st.sidebar.header("🎲 Gerar Dados Aleatórios")
seed = st.sidebar.number_input("Semente Aleatória", min_value=0, value=42)
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
prazo_min = 1
prazo_max = 3
penalidade_min = 3
penalidade_max = 15

# Gerar fluxos aleatórios (arestas)
fluxos = []
for i in setores:
    for j in setores:
        if i != j and i == 'A':  # A só manda
            fluxo = (i, j, np.random.randint(cap_min, cap_max),
                     np.round(np.random.uniform(custo_min, custo_max), 2),
                     np.round(np.random.uniform(juros_min/100, juros_max/100), 4),
                     np.random.randint(prazo_min, prazo_max+1),
                     np.random.randint(penalidade_min, penalidade_max+1))
            fluxos.append(fluxo)

# === Botão para ver dados gerados ===
if st.button("👀 Ver Dados Gerados Aleatoriamente"):
    st.subheader("📊 Demandas por Período e Setor")
    df_demandas = pd.DataFrame([
        {'Período': t, 'Setor': s, 'Demanda': demandas[(t, s)]} 
        for (t, s) in demandas
    ])
    st.dataframe(df_demandas)

    st.subheader("📦 Fluxos Permitidos (Arestas)")
    df_fluxos_aleatorios = pd.DataFrame(fluxos, columns=["De", "Para", "Capacidade", "Custo", "Juros", "Prazo", "Penalidade"])
    st.dataframe(df_fluxos_aleatorios)

# === Resolver o Problema ===
if st.button("🔍 Resolver Otimização"):

    # Modelo
    prob = LpProblem("Fluxo_Caixa_Com_Relaxacao", LpMinimize)
    x = LpVariable.dicts("x", ((i, j, t) for (i, j, _, _, _, _, _) in fluxos for t in periodos), lowBound=0)
    erro = LpVariable.dicts("erro", ((k, t) for k in setores for t in periodos), lowBound=0)
    M = 1000

    custo_total = []
    for (i, j, cap, custo, juros, prazo, penalidade) in fluxos:
        for t in periodos:
            atraso = max(0, t - prazo)
            custo_total.append(custo * x[i, j, t] + juros * x[i, j, t] + penalidade * atraso * x[i, j, t])
    penalidade_erro = lpSum(M * erro[k, t] for k in setores for t in periodos)
    prob += lpSum(custo_total) + penalidade_erro, "Custo_Total_Ajustado"

    for (i, j, cap, _, _, _, _) in fluxos:
        for t in periodos:
            prob += x[i, j, t] <= cap, f"Capacidade_{i}_{j}_t{t}"

    for t in periodos:
        for k in setores:
            entradas = lpSum(x[i, k, t] for (i, j, _, _, _, _, _) in fluxos if j == k and (i, j, t) in x)
            saidas = lpSum(x[k, j, t] for (i, j, _, _, _, _, _) in fluxos if i == k and (i, j, t) in x)
            prob += (entradas - saidas + erro[k, t]) == demandas.get((t, k), 0), f"Balanço_{k}_t{t}"

    prob.solve()

    st.subheader("✅ Resultados da Otimização")
    st.write(f"**Status:** {LpStatus[prob.status]}")
    st.write(f"**Custo Total:** R$ {value(prob.objective):,.2f}")

    # Fluxos e erros
    fluxos_resultado = []
    erros_resultado = []
    for v in prob.variables():
        if "x_" in v.name and v.varValue > 0:
            partes = v.name.split("_")
            de = partes[1].strip("(),'")
            para = partes[2].strip("(),'")
            t = int(partes[3].strip("(),'"))
            fluxos_resultado.append([de, para, t, v.varValue])
        if "erro_" in v.name and v.varValue > 0:
            partes = v.name.split("_")
            setor = partes[1].strip("(),'")
            t = int(partes[2].strip("(),'"))
            erros_resultado.append([setor, t, v.varValue])

    df_fluxos = pd.DataFrame(fluxos_resultado, columns=["De", "Para", "Período", "Fluxo"])
    df_erros = pd.DataFrame(erros_resultado, columns=["Setor", "Período", "Erro"])

    st.write("### 📦 Fluxos Encontrados")
    st.dataframe(df_fluxos)

    st.write("### ⚠️ Demandas Não Atendidas (Erros)")
    st.dataframe(df_erros)

    # Gráficos de Análise
    st.subheader("📊 Análises Gráficas dos Resultados")

    # Gráfico 1: Fluxo Total por Período
    fluxo_por_periodo = df_fluxos.groupby("Período")["Fluxo"].sum()
    fig1, ax1 = plt.subplots()
    fluxo_por_periodo.plot(kind="bar", color="skyblue", ax=ax1)
    ax1.set_title("Fluxo Total por Período")
    ax1.set_ylabel("Valor (R$)")
    st.pyplot(fig1)

    # Gráfico 2: Erros por Setor (caso ocorra)
    if not df_erros.empty:
        erros_por_setor = df_erros.groupby("Setor")["Erro"].sum()
        fig2, ax2 = plt.subplots()
        erros_por_setor.plot(kind="bar", color="salmon", ax=ax2)
        ax2.set_title("Demandas Não Atendidas por Setor")
        ax2.set_ylabel("Valor (R$)")
        st.pyplot(fig2)

    # Grafo
    st.subheader("🌐 Grafo dos Fluxos de Caixa")
    G = nx.DiGraph()
    for s in setores:
        G.add_node(s)
    for _, row in df_fluxos.iterrows():
        i, j, t, f = row
        label = f"R${int(f):,} (t{t})"
        if G.has_edge(i, j):
            G[i][j]['labels'].append(label)
            G[i][j]['weights'].append(f)
        else:
            G.add_edge(i, j, labels=[label], weights=[f])

    pos = {'A': (0, 0)}
    angles = np.linspace(0, 2 * np.pi, len(setores)-1, endpoint=False)
    raio = 3
    for (setor, angle) in zip([s for s in setores if s != 'A'], angles):
        pos[setor] = (raio * np.cos(angle), raio * np.sin(angle))
    edge_labels = { (i, j): "\n".join(G[i][j]['labels']) for i, j in G.edges() }
    edge_widths = [sum(G[i][j]['weights'])/10000 for i, j in G.edges()]
    node_colors = ['#ff9999' if node == 'A' else '#99ccff' for node in G.nodes()]

    fig3, ax3 = plt.subplots(figsize=(10,8))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=edge_widths, arrowsize=25, arrowstyle='-|>', connectionstyle='arc3,rad=0.15')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkred', font_size=10,
                                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5))
    ax3.set_title("🔗 Grafo dos Fluxos de Caixa (Layout Pentagrama)", fontsize=16)
    ax3.axis('off')
    st.pyplot(fig3)

    # Análise final
    st.subheader("📌 Análise Final e Sugestões")
    texto_analise = f"""
- Foram gerados dados aleatórios para demandas e fluxos financeiros.
- A otimização encontrou uma solução **{LpStatus[prob.status]}** com custo total de **R$ {value(prob.objective):,.2f}**.
"""
    if not df_erros.empty:
        texto_analise += f"- Atenção: Existem **{len(df_erros)} casos de demandas não atendidas**, indicando possíveis gargalos.\n"
        texto_analise += "- 💡 Sugestões:\n"
        texto_analise += "  - Aumente as capacidades dos fluxos.\n"
        texto_analise += "  - Reduza as demandas excessivas.\n"
        texto_analise += "  - Avalie prazos e penalidades.\n"
    else:
        texto_analise += "- 🎉 Todas as demandas foram atendidas com sucesso.\n"

    st.markdown(texto_analise)
