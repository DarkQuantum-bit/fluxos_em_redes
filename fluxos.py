import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value

# === Configurações Iniciais ===
st.set_page_config(page_title="MS529 - Otimização de Fluxo de Caixa", layout="wide")
st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>MS529 - Otimização de Fluxo de Caixa</h1>
    <p style='text-align: center; font-size:18px;'>Simule cenários financeiros, visualize fluxos para tomar melhores decisões!</p>
    """,
    unsafe_allow_html=True
)

# Paleta de cores
COR_PRINCIPAL = "#4B8BBE"
COR_DESTAQUE = "#306998"

# === Parâmetros do Modelo ===
setores = ['A', 'B', 'C', 'D', 'E', 'F']
periodos = [1, 2, 3]

st.sidebar.header("⚙️ Parâmetros de Simulação")
seed = st.sidebar.number_input("🔹 Semente Aleatória", min_value=0, value=42)
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

fluxos = []
for i in setores:
    for j in setores:
        if i != j and i == 'A':
            fluxo = (i, j, np.random.randint(cap_min, cap_max),
                     np.round(np.random.uniform(custo_min, custo_max), 2),
                     np.round(np.random.uniform(juros_min/100, juros_max/100), 4),
                     np.random.randint(prazo_min, prazo_max+1),
                     np.random.randint(penalidade_min, penalidade_max+1))
            fluxos.append(fluxo)

# === Botão para ver dados gerados ===
st.markdown("---")
if st.button("📊 Ver Dados Aleatórios Gerados"):
    st.markdown(f"<h3 style='color:{COR_PRINCIPAL};'>📊 Demandas por Período e Setor</h3>", unsafe_allow_html=True)
    df_demandas = pd.DataFrame([
        {'Período': t, 'Setor': s, 'Demanda': demandas[(t, s)]} 
        for (t, s) in demandas
    ])
    st.dataframe(df_demandas)

    st.markdown(f"<h3 style='color:{COR_PRINCIPAL};'>📦 Fluxos Permitidos (Arestas)</h3>", unsafe_allow_html=True)
    df_fluxos_aleatorios = pd.DataFrame(fluxos, columns=["De", "Para", "Capacidade", "Custo", "Juros", "Prazo", "Penalidade"])
    st.dataframe(df_fluxos_aleatorios)

st.markdown("---")

# === Resolver o Problema ===
if st.button("🚀 Resolver Otimização"):
    with st.spinner("⏳ Resolvendo o problema..."):

        prob = LpProblem("Fluxo_Caixa_Com_Relaxacao", LpMinimize)

        # Variáveis de fluxo por arco e período
        x = LpVariable.dicts("x", ((i, j, t) for (i, j, _, _, _, _, _) in fluxos for t in periodos), lowBound=0)
        # Variáveis de erro para demanda relaxada
        erro = LpVariable.dicts("erro", ((k, t) for k in setores for t in periodos), lowBound=0)
        M = 1e6  # Penalidade alta para erro

        # Função objetivo: custo do fluxo + juros + penalidade de atraso + penalização de erro
        custo_total = []
        for (i, j, cap, custo, juros, prazo, penalidade) in fluxos:
            for t in periodos:
                atraso = max(0, t - prazo)
                fluxo_var = x[i, j, t]
                custo_fluxo = custo * fluxo_var
                custo_juros = juros * fluxo_var
                custo_penalidade = penalidade * atraso * fluxo_var
                custo_total.append(custo_fluxo + custo_juros + custo_penalidade)

        penalidade_erro = lpSum(M * erro[k, t] for k in setores for t in periodos)
        prob += lpSum(custo_total) + penalidade_erro

        # Restrição: capacidade máxima do fluxo por arco e período
        for (i, j, cap, _, _, _, _) in fluxos:
            for t in periodos:
                prob += x[i, j, t] <= cap

        # Restrição: balanço de fluxo com erro (relaxação da demanda)
        for t in periodos:
            for k in setores:
                entradas = lpSum(x[i, k, t] for (i, j, _, _, _, _, _) in fluxos if j == k)
                saidas = lpSum(x[k, j, t] for (i, j, _, _, _, _, _) in fluxos if i == k)
                # entradas - saidas + erro == demanda
                prob += entradas - saidas + erro[k, t] == demandas.get((t, k), 0)

        # Resolver o problema
        prob.solve()

    st.markdown(f"<h3 style='color:{COR_PRINCIPAL};'>✅ Resultados da Otimização</h3>", unsafe_allow_html=True)
    st.success(f"Status: {LpStatus[prob.status]} | Custo Total: R$ {value(prob.objective):,.2f}")

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

    with st.expander("📦 Ver Fluxos Encontrados"):
        st.dataframe(df_fluxos)

    with st.expander("⚠️ Demandas Não Atendidas (Erros)"):
        st.dataframe(df_erros)

    # Gráficos de Análise
    st.markdown(f"<h3 style='color:{COR_PRINCIPAL};'>📊 Análises Gráficas</h3>", unsafe_allow_html=True)
    fluxo_por_periodo = df_fluxos.groupby("Período")["Fluxo"].sum()
    fig1, ax1 = plt.subplots()
    fluxo_por_periodo.plot(kind="bar", color=COR_PRINCIPAL, ax=ax1)
    ax1.set_title("Fluxo Total por Período", fontsize=14)
    ax1.set_ylabel("Valor (R$)")
    st.pyplot(fig1)

    if not df_erros.empty:
        erros_por_setor = df_erros.groupby("Setor")["Erro"].sum()
        fig2, ax2 = plt.subplots()
        erros_por_setor.plot(kind="bar", color="salmon", ax=ax2)
        ax2.set_title("Demandas Não Atendidas por Setor", fontsize=14)
        ax2.set_ylabel("Valor (R$)")
        st.pyplot(fig2)

    # 🌐 Grafo dos Fluxos de Caixa - Melhorado
    st.markdown(f"<h3 style='color:{COR_PRINCIPAL};'>🌐 Grafo dos Fluxos de Caixa</h3>", unsafe_allow_html=True)

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

    pos = {}
    raio = 5
    pos['A'] = (0, 0)
    angles = np.linspace(0, 2 * np.pi, len(setores) - 1, endpoint=False)
    for (setor, angle) in zip([s for s in setores if s != 'A'], angles):
        pos[setor] = (raio * np.cos(angle), raio * np.sin(angle))

    edge_labels = {(i, j): "\n".join(G[i][j]['labels']) for i, j in G.edges()}
    edge_widths = [0.5 + sum(G[i][j]['weights']) / 50000 for i, j in G.edges()]
    node_colors = ['#FF8C00' if node == 'A' else '#1E90FF' for node in G.nodes()]

    fig3, ax3 = plt.subplots(figsize=(10, 10), facecolor="#1a1a1a")
    ax3.set_facecolor("#1a1a1a")

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2500, edgecolors='white', linewidths=1.5, ax=ax3)
    nx.draw_networkx_labels(G, pos, font_size=16, font_weight='bold', font_color='white', ax=ax3)
    nx.draw_networkx_edges(G, pos, width=edge_widths, arrowsize=25, arrowstyle='-|>', connectionstyle='arc3,rad=0.2', edge_color='white', alpha=0.8, ax=ax3)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='salmon', font_size=10,
                                 bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="white", lw=0.5, alpha=0.7), ax=ax3)

    ax3.set_title("Grafo dos Fluxos de Caixa", fontsize=16, color='white', pad=20)
    ax3.axis('off')
    st.pyplot(fig3)

    # 📌 Análise Final
    st.markdown(f"<h3 style='color:{COR_PRINCIPAL};'>📌 Análise Final e Sugestões</h3>", unsafe_allow_html=True)
    texto_analise = f"""
- Foram gerados dados aleatórios para demandas e fluxos financeiros.
- A otimização encontrou uma solução **{LpStatus[prob.status]}** com custo total de **R$ {value(prob.objective):,.2f}**.
"""
    if not df_erros.empty:
        texto_analise += f"- ⚠️ Existem **{len(df_erros)} casos de demandas não atendidas**, indicando possíveis gargalos.\n"
        texto_analise += "- 💡 **Sugestões**:\n"
        texto_analise += "  - Aumente as capacidades dos fluxos.\n"
        texto_analise += "  - Reduza as demandas excessivas.\n"
        texto_analise += "  - Avalie prazos e penalidades.\n"
    else:
        texto_analise += "- 🎉 Todas as demandas foram atendidas com sucesso.\n"

    st.markdown(texto_analise)
