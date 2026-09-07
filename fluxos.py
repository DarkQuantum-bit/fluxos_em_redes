import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value

# Configurações iniciais
st.set_page_config(page_title="Otimização de Fluxo de Caixa", layout="wide")

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f3a5f;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #4a6fa5;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #1f3a5f;
        color: white;
        font-weight: 600;
        border-radius: 4px;
    }
    .stButton button:hover {
        background-color: #4a6fa5;
    }
    .sidebar-content {
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

COR_PRINCIPAL = "#1f3a5f"
COR_SECUNDARIA = "#4a6fa5"

# Definições de setores e períodos
setores = ['A', 'B', 'C', 'D', 'E', 'F']
periodos = [1, 2, 3]

# Estado da sessão
if 'modo_dados' not in st.session_state:
    st.session_state.modo_dados = "Gerar aleatoriamente"
if 'resultados' not in st.session_state:
    st.session_state.resultados = None

def validar_dados(demandas, fluxos):
    """Valida se os dados inseridos são consistentes"""
    erros = []
    warnings = []
    
    # Verificar equilíbrio de demandas
    for t in periodos:
        total_positivo = sum(demandas[(t, s)] for s in setores if demandas[(t, s)] > 0)
        total_negativo = sum(demandas[(t, s)] for s in setores if demandas[(t, s)] < 0)
        if abs(total_positivo + total_negativo) > 0.01:
            erros.append(f"Período {t}: Demandas não equilibradas (diferença de R$ {abs(total_positivo + total_negativo):,.2f})")
    
    # Verificar capacidades
    for (i, j, cap, custo, juros) in fluxos:
        if cap <= 0:
            erros.append(f"Fluxo {i}->{j}: Capacidade deve ser positiva")
        if custo < 0 or juros < 0:
            erros.append(f"Fluxo {i}->{j}: Custos e juros não podem ser negativos")
    
    # Verificar se há capacidade suficiente
    total_demanda = sum(max(0, demandas[(t, s)]) for t in periodos for s in setores if s != 'A')
    total_capacidade = sum(cap for (_, _, cap, _, _) in fluxos if _ != 'A')
    if total_capacidade < total_demanda:
        warnings.append(f"Capacidade total ({total_capacidade:,.0f}) pode ser insuficiente para demanda total ({total_demanda:,.0f})")
    
    return erros, warnings

def criar_modelo_otimizacao(demandas, fluxos, modo, M=10.0):
    """Cria e resolve o modelo de otimização"""
    prob = LpProblem(f"Fluxo_Caixa_{modo}", LpMinimize)
    
    # Variáveis de decisão
    x = LpVariable.dicts("x", ((i, j, t) for (i, j, _, _, _) in fluxos for t in periodos), lowBound=0)
    saldo = LpVariable.dicts("saldo", ((s, t) for s in setores for t in periodos), lowBound=0)
    deficit = LpVariable.dicts("deficit", ((s, t) for s in setores for t in periodos), lowBound=0)
    
    # Variáveis de erro (se relaxado)
    if modo == "Com relaxamento":
        erro_pos = LpVariable.dicts("erro_pos", ((s, t) for s in setores for t in periodos), lowBound=0)
        erro_neg = LpVariable.dicts("erro_neg", ((s, t) for s in setores for t in periodos), lowBound=0)
    
    # Função objetivo
    custo_total = lpSum((custo + juros) * x[i, j, t] 
                        for (i, j, _, custo, juros) in fluxos 
                        for t in periodos)
    
    # Custo de oportunidade do saldo
    taxa_oportunidade = 0.02
    custo_oportunidade = lpSum(taxa_oportunidade * saldo[s, t] 
                               for s in setores 
                               for t in periodos)
    
    # Custo de juros sobre déficit
    taxa_juros_deficit = 0.05
    custo_juros = lpSum(taxa_juros_deficit * deficit[s, t] 
                        for s in setores 
                        for t in periodos)
    
    prob += custo_total + custo_oportunidade + custo_juros
    
    if modo == "Com relaxamento":
        penalidade = lpSum(M * (erro_pos[s, t] + erro_neg[s, t])
                          for s in setores 
                          for t in periodos)
        prob += penalidade
    
    # Restrições de capacidade
    for (i, j, cap, _, _) in fluxos:
        for t in periodos:
            prob += x[i, j, t] <= cap, f"Capacidade_{i}_{j}_{t}"
    
    # Restrições de balanço
    for s in setores:
        for t in periodos:
            entradas = lpSum(x[i, s, t] for (i, j, _, _, _) in fluxos if j == s)
            saidas = lpSum(x[s, j, t] for (i, j, _, _, _) in fluxos if i == s)
            saldo_prev = 0 if t == 1 else saldo[s, t-1]
            
            if modo == "Com relaxamento":
                prob += (entradas - saidas + saldo_prev + erro_pos[s, t] - erro_neg[s, t]
                        == demandas.get((t, s), 0) + saldo[s, t],
                        f"Balanco_{s}_{t}")
            else:
                prob += (entradas - saidas + saldo_prev
                        == demandas.get((t, s), 0) + saldo[s, t],
                        f"Balanco_{s}_{t}")
    
    # Restrições de déficit
    for s in setores:
        for t in periodos:
            prob += deficit[s, t] >= -saldo[s, t], f"Deficit_{s}_{t}"
    
    # Evitar fluxos circulares para o setor fornecedor A
    for (i, j, _, _, _) in fluxos:
        if j == 'A':
            for t in periodos:
                prob += x[i, j, t] == 0, f"Sem_entrada_A_{i}_{t}"
    
    prob.solve()
    return prob

def extrair_resultados(prob, modo):
    """Extrai resultados do modelo resolvido"""
    fluxos_resultado = []
    erros_resultado = []
    saldos_resultado = []
    
    for v in prob.variables():
        if v.varValue > 0:
            if "x_" in v.name:
                partes = v.name.split("_")
                de = partes[1].strip("(),' ")
                para = partes[2].strip("(),' ")
                t = int(partes[3].strip("(),' "))
                fluxos_resultado.append([de, para, t, v.varValue])
            elif "erro_pos_" in v.name or "erro_neg_" in v.name:
                partes = v.name.split("_")
                tipo = "positivo" if "pos" in v.name else "negativo"
                setor = partes[2].strip("(),' ")
                t = int(partes[3].strip("(),' "))
                erros_resultado.append([setor, t, v.varValue, tipo])
            elif "saldo_" in v.name:
                partes = v.name.split("_")
                setor = partes[1].strip("(),' ")
                t = int(partes[2].strip("(),' "))
                saldos_resultado.append([setor, t, v.varValue])
    
    return fluxos_resultado, erros_resultado, saldos_resultado

def criar_grafo_interativo(df_fluxos, modo):
    """Cria grafo interativo com Plotly"""
    G = nx.DiGraph()
    for s in setores:
        G.add_node(s)
    
    for _, row in df_fluxos.iterrows():
        if G.has_edge(row['De'], row['Para']):
            G[row['De']][row['Para']]['weight'] += row['Fluxo']
        else:
            G.add_edge(row['De'], row['Para'], weight=row['Fluxo'])
    
    pos = {'A': (0, 0)}
    for idx, s in enumerate(['B', 'C', 'D', 'E', 'F']):
        angle = 2 * math.pi * idx / 5
        pos[s] = (5 * np.cos(angle), 5 * np.sin(angle))
    
    edge_traces = []
    for (i, j) in G.edges():
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=min(G[i][j]['weight']/10000, 8), color='#7f8c8d'),
            hoverinfo='text',
            text=f'{i} para {j}<br>Fluxo total: R$ {G[i][j]["weight"]:,.2f}',
            mode='lines'
        ))
    
    node_trace = go.Scatter(
        x=[pos[s][0] for s in setores],
        y=[pos[s][1] for s in setores],
        mode='markers+text',
        text=list(setores),
        textposition='middle center',
        marker=dict(size=40, color=COR_PRINCIPAL),
        textfont=dict(color='white', size=16),
        hoverinfo='text',
        hovertext=[f'Setor {s}' for s in setores]
    )
    
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=f'Grafo de Fluxos - {modo}',
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def criar_grafico_comparativo(fluxos_resultado):
    """Cria gráfico comparativo por período"""
    fluxos_por_periodo = {t: 0 for t in periodos}
    for _, row in pd.DataFrame(fluxos_resultado, columns=["De", "Para", "Período", "Fluxo"]).iterrows():
        fluxos_por_periodo[row['Período']] += row['Fluxo']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(periodos, [fluxos_por_periodo[t] for t in periodos], 
                  color=COR_SECUNDARIA, alpha=0.8)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'R$ {height:,.0f}',
                ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Período', fontsize=12)
    ax.set_ylabel('Volume de Transações (R$)', fontsize=12)
    ax.set_title('Volume de Transações por Período', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(periodos)
    
    return fig

# Interface principal
st.markdown('<p class="main-header">Otimização de Fluxo de Caixa</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistema de apoio à decisão para alocação otimizada de recursos financeiros entre setores</p>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("Configurações")
    
    # Modo de dados
    if st.button("Alternar modo de dados"):
        st.session_state.modo_dados = "Inserir manualmente" if st.session_state.modo_dados == "Gerar aleatoriamente" else "Gerar aleatoriamente"
    
    modo_dados = st.session_state.modo_dados
    st.write(f"Modo atual: **{modo_dados}**")
    
    # Cenários
    st.subheader("Cenários de Análise")
    cenario = st.selectbox("Selecione o cenário", ["Padrão", "Otimista", "Pessimista"])
    
    # Fator de ajuste
    if cenario == "Otimista":
        fator_ajuste = 0.8
        st.info("Cenário otimista: custos 20% menores")
    elif cenario == "Pessimista":
        fator_ajuste = 1.2
        st.info("Cenário pessimista: custos 20% maiores")
    else:
        fator_ajuste = 1.0
        st.info("Cenário padrão: custos normais")
    
    # Parâmetros de penalização
    st.subheader("Parâmetros de Otimização")
    M = st.number_input("Penalização por erro (M)", value=10.0, min_value=1.0, max_value=100.0)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Inicialização das estruturas de dados
demandas = {}
fluxos = []

# Modo de geração aleatória
if modo_dados == "Gerar aleatoriamente":
    with st.sidebar:
        seed = st.number_input("Seed aleatória", min_value=0, value=42)
        np.random.seed(seed)
        
        st.subheader("Parâmetros de Geração")
        demanda_total = st.number_input("Demanda total por período (R$)", value=400000, step=10000)
        cap_min = st.number_input("Capacidade mínima (R$)", value=30000, step=1000)
        cap_max = st.number_input("Capacidade máxima (R$)", value=120000, step=1000)
        custo_min = st.number_input("Custo unitário mínimo", value=1.0, step=0.1)
        custo_max = st.number_input("Custo unitário máximo", value=3.0, step=0.1)
        juros_min = st.number_input("Juros mínimo (%)", value=1.0, step=0.1)
        juros_max = st.number_input("Juros máximo (%)", value=5.0, step=0.1)
    
    for t in periodos:
        proporcoes = np.random.dirichlet(np.ones(len(setores) - 1), 1).flatten()
        for idx, s in enumerate([x for x in setores if x != 'A']):
            demandas[(t, s)] = int(demanda_total * proporcoes[idx])
        demandas[(t, 'A')] = -sum(demandas[(t, s)] for s in setores if s != 'A')
    
    for i in setores:
        for j in setores:
            if i != j and i != 'A':  # A não envia fluxos
                cap = np.random.randint(cap_min, cap_max)
                custo = np.round(np.random.uniform(custo_min, custo_max), 2)
                juros = np.round(np.random.uniform(juros_min / 100, juros_max / 100), 4)
                fluxos.append((i, j, cap, custo * fator_ajuste, juros * fator_ajuste))

# Modo de inserção manual
else:
    st.subheader("Inserir dados manualmente")
    
    tab_demandas, tab_fluxos = st.tabs(["Demandas", "Fluxos"])
    
    with tab_demandas:
        for t in periodos:
            st.markdown(f"**Período {t}**")
            cols = st.columns(len(setores))
            for idx, s in enumerate(setores):
                with cols[idx]:
                    demandas[(t, s)] = st.number_input(f"Setor {s} (P{t})", value=0, step=1000, format="%d")
    
    with tab_fluxos:
        for i in setores:
            for j in setores:
                if i != j and i != 'A':
                    with st.expander(f"Fluxo de {i} para {j}"):
                        cols = st.columns(3)
                        with cols[0]:
                            cap = st.number_input(f"Capacidade {i}->{j}", value=50000, step=1000)
                        with cols[1]:
                            custo = st.number_input(f"Custo {i}->{j}", value=2.0, step=0.1, format="%.2f")
                        with cols[2]:
                            juros = st.number_input(f"Juros (%) {i}->{j}", value=3.0, step=0.1) / 100
                        fluxos.append((i, j, cap, custo * fator_ajuste, juros * fator_ajuste))

# Validação dos dados
erros_validacao, warnings_validacao = validar_dados(demandas, fluxos)

if erros_validacao:
    st.error("Erros de validação encontrados:")
    for erro in erros_validacao:
        st.error(f"• {erro}")
elif warnings_validacao:
    st.warning("Avisos:")
    for warning in warnings_validacao:
        st.warning(f"• {warning}")

# Exibição dos dados
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demandas por Período e Setor")
    df_demandas = pd.DataFrame([
        {'Período': t, 'Setor': s, 'Demanda': demandas[(t, s)]} 
        for (t, s) in demandas
    ])
    st.dataframe(df_demandas, use_container_width=True)

with col2:
    st.subheader("Fluxos Permitidos")
    df_fluxos = pd.DataFrame(fluxos, columns=["De", "Para", "Capacidade", "Custo", "Juros"])
    df_fluxos['Custo'] = df_fluxos['Custo'].round(2)
    df_fluxos['Juros'] = df_fluxos['Juros'].round(4)
    st.dataframe(df_fluxos, use_container_width=True)

# Botão de otimização
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
with col_btn2:
    botao_otimizar = st.button("Executar Otimização", type="primary", use_container_width=True)

# Botão para mostrar modelagem
with st.sidebar:
    if st.button("Mostrar Modelagem Matemática"):
        st.session_state.mostrar_modelagem = True

if 'mostrar_modelagem' in st.session_state and st.session_state.mostrar_modelagem:
    with st.expander("Modelagem Matemática", expanded=True):
        st.markdown(r"""
### Formulação Matemática do Problema

**Variáveis de Decisão:**
- $x_{ijt} \geq 0$: Fluxo financeiro do setor $i$ para $j$ no período $t$
- $s_{st} \geq 0$: Saldo acumulado do setor $s$ no período $t$  
- $d_{st} \geq 0$: Déficit financeiro do setor $s$ no período $t$
- $e^+_{st}, e^-_{st} \geq 0$: Violações de demanda (modo relaxado)

**Função Objetivo:**
$$\min \sum_{i,j,t} (c_{ij} + r_{ij})x_{ijt} + \alpha\sum_{s,t} s_{st} + \beta\sum_{s,t} d_{st} + M\sum_{s,t}(e^+_{st} + e^-_{st})$$

**Sujeito a:**
1. **Capacidade:** $x_{ijt} \leq cap_{ij}, \forall i,j,t$
2. **Balanço de fluxo:** $\sum_i x_{ist} - \sum_j x_{sjt} + s_{s,t-1} = D_{st} + s_{st}, \forall s,t$
3. **Déficit:** $d_{st} \geq -s_{st}, \forall s,t$
4. **Não negatividade:** $x_{ijt}, s_{st}, d_{st} \geq 0$

**Onde:**
- $c_{ij}$: Custo unitário do fluxo de $i$ para $j$
- $r_{ij}$: Taxa de juros do fluxo de $i$ para $j$
- $\alpha$: Taxa de custo de oportunidade do saldo
- $\beta$: Taxa de juros sobre déficit
- $M$: Penalização por violação de demanda
""")

if botao_otimizar:
    if not fluxos:
        st.error("Nenhum fluxo definido. Configure os fluxos permitidos antes de otimizar.")
    else:
        resultados = {}
        for modo in ["Sem relaxamento", "Com relaxamento"]:
            st.markdown(f"## Resultados - {modo}")
            
            with st.spinner(f"Resolvendo problema {modo}..."):
                prob = criar_modelo_otimizacao(demandas, fluxos, modo, M)
                
                # Extrair resultados
                fluxos_resultado, erros_resultado, saldos_resultado = extrair_resultados(prob, modo)
                
                # Armazenar resultados
                resultados[modo] = {
                    'status': LpStatus[prob.status],
                    'custo_total': value(prob.objective),
                    'fluxos': fluxos_resultado,
                    'erros': erros_resultado,
                    'saldos': saldos_resultado
                }
                
                # Exibir métricas principais
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Status", LpStatus[prob.status])
                
                with col2:
                    st.metric("Custo Total", f"R$ {value(prob.objective):,.2f}")
                
                with col3:
                    total_fluxo = sum(f[3] for f in fluxos_resultado)
                    st.metric("Volume Total Transacionado", f"R$ {total_fluxo:,.2f}")
                
                # Análise de sensibilidade
                if fluxos_resultado:
                    st.markdown("### Análise de Utilização de Capacidade")
                    df_fluxos_resultado = pd.DataFrame(fluxos_resultado, 
                                                       columns=["De", "Para", "Período", "Fluxo"])
                    
                    # Calcular utilização
                    utilizacao_data = []
                    for (i, j, cap, _, _) in fluxos:
                        fluxo_total = df_fluxos_resultado[
                            (df_fluxos_resultado['De'] == i) & 
                            (df_fluxos_resultado['Para'] == j)
                        ]['Fluxo'].sum()
                        utilizacao = (fluxo_total / (cap * len(periodos))) * 100
                        utilizacao_data.append({
                            'De': i, 'Para': j,
                            'Capacidade': cap,
                            'Fluxo Total': fluxo_total,
                            'Utilização (%)': round(utilizacao, 1)
                        })
                    
                    df_utilizacao = pd.DataFrame(utilizacao_data)
                    df_utilizacao = df_utilizacao[df_utilizacao['Fluxo Total'] > 0]
                    
                    if not df_utilizacao.empty:
                        st.dataframe(df_utilizacao, use_container_width=True)
                        
                        # Gráfico de utilização
                        fig_utilizacao, ax_utilizacao = plt.subplots(figsize=(10, 5))
                        bars = ax_utilizacao.bar(
                            [f"{row['De']}->{row['Para']}" for _, row in df_utilizacao.iterrows()],
                            df_utilizacao['Utilização (%)'],
                            color=COR_SECUNDARIA, alpha=0.8
                        )
                        ax_utilizacao.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Capacidade máxima')
                        ax_utilizacao.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Alta utilização')
                        ax_utilizacao.set_xlabel('Fluxos', fontsize=12)
                        ax_utilizacao.set_ylabel('Utilização (%)', fontsize=12)
                        ax_utilizacao.set_title('Utilização de Capacidade por Fluxo', fontsize=14, fontweight='bold')
                        ax_utilizacao.legend()
                        ax_utilizacao.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        st.pyplot(fig_utilizacao)
                
                # Exibir fluxos
                st.markdown("### Fluxos Otimizados")
                if fluxos_resultado:
                    df_fluxos_resultado = pd.DataFrame(fluxos_resultado, 
                                                       columns=["De", "Para", "Período", "Fluxo"])
                    df_fluxos_resultado['Fluxo'] = df_fluxos_resultado['Fluxo'].round(2)
                    st.dataframe(df_fluxos_resultado, use_container_width=True)
                else:
                    st.info("Nenhum fluxo otimizado encontrado.")
                
                # Exibir erros
                if erros_resultado:
                    st.markdown("### Demandas Não Atendidas")
                    df_erros = pd.DataFrame(erros_resultado, 
                                           columns=["Setor", "Período", "Erro", "Tipo"])
                    st.dataframe(df_erros, use_container_width=True)
                
                # Exibir saldos
                if saldos_resultado:
                    st.markdown("### Saldos por Setor e Período")
                    df_saldos = pd.DataFrame(saldos_resultado, 
                                            columns=["Setor", "Período", "Saldo"])
                    df_saldos['Saldo'] = df_saldos['Saldo'].round(2)
                    
                    # Pivot table
                    pivot_saldos = df_saldos.pivot(index='Setor', columns='Período', values='Saldo')
                    st.dataframe(pivot_saldos, use_container_width=True)
                
                # Gráficos comparativos
                if fluxos_resultado:
                    st.markdown("### Análise Temporal")
                    fig_comparativo = criar_grafico_comparativo(fluxos_resultado)
                    st.pyplot(fig_comparativo)
                
                # Grafo interativo
                if fluxos_resultado:
                    st.markdown("### Visualização de Rede")
                    fig_grafo = criar_grafo_interativo(pd.DataFrame(fluxos_resultado, 
                                                                   columns=["De", "Para", "Período", "Fluxo"]), 
                                                      modo)
                    st.plotly_chart(fig_grafo, use_container_width=True)
            
            st.markdown("---")
        
        # Comparação entre modos
        if len(resultados) == 2:
            st.markdown("## Comparação entre Modos")
            
            col1, col2 = st.columns(2)
            
            for idx, (modo, res) in enumerate(resultados.items()):
                with (col1 if idx == 0 else col2):
                    st.markdown(f"### {modo}")
                    st.markdown(f"**Status:** {res['status']}")
                    st.markdown(f"**Custo Total:** R$ {res['custo_total']:,.2f}")
                    st.markdown(f"**Número de Fluxos:** {len(res['fluxos'])}")
                    st.markdown(f"**Erros:** {len(res['erros'])}")
            
            # Gráfico comparativo de custos
            fig_comparacao, ax_comparacao = plt.subplots(figsize=(8, 5))
            custos = [res['custo_total'] for res in resultados.values()]
            modos = list(resultados.keys())
            bars = ax_comparacao.bar(modos, custos, color=[COR_PRINCIPAL, COR_SECUNDARIA], alpha=0.8)
            
            for bar in bars:
                height = bar.get_height()
                ax_comparacao.text(bar.get_x() + bar.get_width()/2., height,
                                  f'R$ {height:,.2f}',
                                  ha='center', va='bottom', fontsize=10)
            
            ax_comparacao.set_ylabel('Custo Total (R$)', fontsize=12)
            ax_comparacao.set_title('Comparação de Custos entre Modos', fontsize=14, fontweight='bold')
            ax_comparacao.grid(True, alpha=0.3)
            st.pyplot(fig_comparacao)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Desenvolvido para a disciplina MS529 - Otimização de Fluxo de Caixa</p>",
    unsafe_allow_html=True
)
