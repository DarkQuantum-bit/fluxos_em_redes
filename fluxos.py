import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import math
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value

# Configurações iniciais
st.set_page_config(page_title="Otimização de Fluxo de Caixa", layout="wide", initial_sidebar_state="expanded")

# CSS personalizado para dashboard
st.markdown("""
<style>
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #1f3a5f 0%, #2c5282 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Cards de métricas */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1f3a5f;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f3a5f;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Seções */
    .section-header {
        color: #1f3a5f;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* DataFrames */
    .dataframe {
        font-size: 0.9rem !important;
    }
    
    /* Botões */
    .stButton button {
        background: linear-gradient(135deg, #1f3a5f 0%, #2c5282 100%);
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar */
    .sidebar-content {
        padding: 1rem 0;
    }
    .sidebar-header {
        color: #1f3a5f;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

COR_PRINCIPAL = "#1f3a5f"
COR_SECUNDARIA = "#4a6fa5"
COR_TERCIARIA = "#48bb78"

# Definições de setores e períodos
setores = ['A', 'B', 'C', 'D', 'E', 'F']
periodos = [1, 2, 3]

# Estado da sessão
if 'modo_dados' not in st.session_state:
    st.session_state.modo_dados = "Gerar aleatoriamente"
if 'resultados' not in st.session_state:
    st.session_state.resultados = None
if 'dados_carregados' not in st.session_state:
    st.session_state.dados_carregados = False

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

def criar_grafo_direcionado_temporal(df_fluxos, modo):
    """Cria grafo direcionado com informações temporais usando Plotly"""
    if df_fluxos.empty:
        return None
    
    # Preparar dados para o grafo
    edges_data = []
    for _, row in df_fluxos.iterrows():
        edges_data.append({
            'source': row['De'],
            'target': row['Para'],
            'periodo': row['Período'],
            'fluxo': row['Fluxo']
        })
    
    # Agregar fluxos por aresta (soma total)
    df_agregado = df_fluxos.groupby(['De', 'Para'])['Fluxo'].sum().reset_index()
    
    # Criar posições dos nós
    pos = {'A': (0, 0)}
    for idx, s in enumerate(['B', 'C', 'D', 'E', 'F']):
        angle = 2 * math.pi * idx / 5
        pos[s] = (5 * np.cos(angle), 5 * np.sin(angle))
    
    # Criar traços para arestas com setas
    edge_traces = []
    
    for _, edge in df_agregado.iterrows():
        source, target, fluxo_total = edge['De'], edge['Para'], edge['Fluxo']
        
        # Filtrar fluxos por período para essa aresta
        fluxos_periodo = df_fluxos[
            (df_fluxos['De'] == source) & 
            (df_fluxos['Para'] == target)
        ]
        
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        
        # Criar hover text com informações por período
        hover_text = f"{source} → {target}<br>Fluxo Total: R$ {fluxo_total:,.2f}<br><br>"
        for _, fluxo_periodo in fluxos_periodo.iterrows():
            hover_text += f"Período {fluxo_periodo['Período']}: R$ {fluxo_periodo['Fluxo']:,.2f}<br>"
        
        # Calcular curvatura para evitar sobreposição
        dx = x1 - x0
        dy = y1 - y0
        dist = math.sqrt(dx**2 + dy**2)
        
        # Criar curva suave
        t = np.linspace(0, 1, 20)
        curve_x = x0 + dx * t
        curve_y = y0 + dy * t
        
        # Adicionar curvatura para arestas bidirecionais
        if df_agregado[(df_agregado['De'] == target) & (df_agregado['Para'] == source)].shape[0] > 0:
            offset = 0.5
            curve_x = x0 + dx * t + offset * np.sin(np.pi * t) * (dy / dist if dist > 0 else 0)
            curve_y = y0 + dy * t - offset * np.sin(np.pi * t) * (dx / dist if dist > 0 else 0)
        
        edge_traces.append(go.Scatter(
            x=curve_x,
            y=curve_y,
            mode='lines+markers',
            line=dict(
                width=min(fluxo_total / 50000, 6),
                color=COR_SECUNDARIA,
                dash='solid'
            ),
            marker=dict(
                size=6,
                symbol='arrow',
                angleref='previous'
            ),
            hoverinfo='text',
            text=hover_text,
            name=f"{source}→{target}"
        ))
    
    # Criar traço para nós
    node_trace = go.Scatter(
        x=[pos[s][0] for s in setores],
        y=[pos[s][1] for s in setores],
        mode='markers+text',
        text=list(setores),
        textposition='middle center',
        marker=dict(
            size=50,
            color=COR_PRINCIPAL,
            line=dict(color='white', width=2)
        ),
        textfont=dict(color='white', size=20, family='Arial Black'),
        hoverinfo='text',
        hovertext=[f"Setor {s}" for s in setores],
        name='Setores'
    )
    
    # Criar figura
    fig = go.Figure(data=edge_traces + [node_trace])
    
    # Atualizar layout
    fig.update_layout(
        title=dict(
            text=f'Fluxos Direcionados - {modo}',
            font=dict(size=20, color=COR_PRINCIPAL)
        ),
        showlegend=False,
        hovermode='closest',
        xaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-7, 7]
        ),
        yaxis=dict(
            showgrid=False, 
            zeroline=False, 
            showticklabels=False,
            range=[-7, 7]
        ),
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='white'
    )
    
    return fig

def criar_grafico_sankey(df_fluxos, modo):
    """Cria diagrama de Sankey para visualizar fluxos"""
    if df_fluxos.empty:
        return None
    
    # Preparar dados para Sankey
    labels = list(setores)
    source = []
    target = []
    values = []
    
    for _, row in df_fluxos.iterrows():
        source.append(labels.index(row['De']))
        target.append(labels.index(row['Para']))
        values.append(row['Fluxo'])
    
    # Criar figura
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=COR_PRINCIPAL
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            color=COR_SECUNDARIA
        )
    )])
    
    fig.update_layout(
        title=dict(
            text=f'Diagrama de Fluxo - {modo}',
            font=dict(size=20, color=COR_PRINCIPAL)
        ),
        font=dict(size=12),
        height=500
    )
    
    return fig

# Interface principal
st.markdown("""
<div class="main-header">
    <h1>Otimização de Fluxo de Caixa</h1>
    <p>Dashboard para análise e otimização de alocação de recursos entre setores</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-header">CONFIGURAÇÕES</p>', unsafe_allow_html=True)
    
    # Modo de dados
    modo_dados = st.radio(
        "Modo de Entrada de Dados",
        ["Gerar aleatoriamente", "Inserir manualmente"],
        horizontal=False
    )
    st.session_state.modo_dados = modo_dados
    
    st.divider()
    
    # Cenários
    st.markdown('<p class="sidebar-header">CENÁRIOS DE ANÁLISE</p>', unsafe_allow_html=True)
    cenario = st.selectbox(
        "Selecione o cenário",
        ["Padrão", "Otimista", "Pessimista"],
        help="Ajusta os custos conforme o cenário econômico"
    )
    
    # Fator de ajuste
    if cenario == "Otimista":
        fator_ajuste = 0.8
        st.caption("Cenário otimista: custos 20% menores")
    elif cenario == "Pessimista":
        fator_ajuste = 1.2
        st.caption("Cenário pessimista: custos 20% maiores")
    else:
        fator_ajuste = 1.0
        st.caption("Cenário padrão: custos normais")
    
    st.divider()
    
    # Parâmetros de otimização
    st.markdown('<p class="sidebar-header">PARÂMETROS DE OTIMIZAÇÃO</p>', unsafe_allow_html=True)
    M = st.number_input(
        "Penalização por erro (M)",
        value=10.0,
        min_value=1.0,
        max_value=100.0,
        step=1.0,
        help="Valor da penalização aplicada às violações de demanda"
    )
    
    taxa_oportunidade = st.number_input(
        "Taxa de custo de oportunidade (%)",
        value=2.0,
        min_value=0.0,
        max_value=10.0,
        step=0.5,
        help="Custo de manter saldo positivo"
    ) / 100
    
    st.divider()
    
    # Botão para mostrar modelagem
    if st.button("Mostrar Modelagem Matemática", use_container_width=True):
        st.session_state.mostrar_modelagem = not st.session_state.get('mostrar_modelagem', False)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Área principal
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

# Inicialização das estruturas de dados
demandas = {}
fluxos = []

# Modo de geração aleatória
if modo_dados == "Gerar aleatoriamente":
    st.markdown("### Configuração de Dados Aleatórios")
    
    with st.expander("Parâmetros de Geração", expanded=True):
        col_params1, col_params2, col_params3 = st.columns(3)
        
        with col_params1:
            seed = st.number_input("Seed aleatória", min_value=0, value=42, help="Para reprodutibilidade dos dados")
            demanda_total = st.number_input("Demanda total por período (R$)", value=400000, step=10000, format="%d")
        
        with col_params2:
            cap_min = st.number_input("Capacidade mínima (R$)", value=30000, step=1000, format="%d")
            cap_max = st.number_input("Capacidade máxima (R$)", value=120000, step=1000, format="%d")
        
        with col_params3:
            custo_min = st.number_input("Custo unitário mínimo", value=1.0, step=0.1, format="%.2f")
            custo_max = st.number_input("Custo unitário máximo", value=3.0, step=0.1, format="%.2f")
        
        col_params4, col_params5 = st.columns(2)
        with col_params4:
            juros_min = st.number_input("Juros mínimo (%)", value=1.0, step=0.1, format="%.2f")
        with col_params5:
            juros_max = st.number_input("Juros máximo (%)", value=5.0, step=0.1, format="%.2f")
    
    np.random.seed(seed)
    
    for t in periodos:
        proporcoes = np.random.dirichlet(np.ones(len(setores) - 1), 1).flatten()
        for idx, s in enumerate([x for x in setores if x != 'A']):
            demandas[(t, s)] = int(demanda_total * proporcoes[idx])
        demandas[(t, 'A')] = -sum(demandas[(t, s)] for s in setores if s != 'A')
    
    for i in setores:
        for j in setores:
            if i != j and i != 'A':
                cap = np.random.randint(cap_min, cap_max)
                custo = np.round(np.random.uniform(custo_min, custo_max), 2)
                juros = np.round(np.random.uniform(juros_min / 100, juros_max / 100), 4)
                fluxos.append((i, j, cap, custo * fator_ajuste, juros * fator_ajuste))

# Modo de inserção manual
else:
    st.markdown("### Configuração de Dados Manuais")
    
    tab_demandas, tab_fluxos = st.tabs(["Demandas", "Fluxos Permitidos"])
    
    with tab_demandas:
        st.markdown("#### Defina as demandas de cada setor por período")
        
        # Criar dataframe editável
        df_demandas_input = pd.DataFrame(
            index=setores,
            columns=[f"Período {t}" for t in periodos]
        )
        df_demandas_input.index.name = "Setor"
        
        # Preencher com valores existentes ou zeros
        for s in setores:
            for t in periodos:
                df_demandas_input.loc[s, f"Período {t}"] = demandas.get((t, s), 0)
        
        # Editor de dados
        df_demandas_editado = st.data_editor(
            df_demandas_input,
            use_container_width=True,
            num_rows="fixed",
            hide_index=False,
            column_config={
                "Setor": st.column_config.TextColumn("Setor", disabled=True),
                **{f"Período {t}": st.column_config.NumberColumn(
                    f"Período {t}",
                    min_value=-1000000,
                    max_value=1000000,
                    step=1000,
                    format="R$ %d"
                ) for t in periodos}
            }
        )
        
        # Atualizar demandas
        for s in setores:
            for t in periodos:
                demandas[(t, s)] = float(df_demandas_editado.loc[s, f"Período {t}"])
        
        st.info("Nota: O Setor A deve ter demanda negativa (fornecedor) e os demais setores demanda positiva (consumidores).")
    
    with tab_fluxos:
        st.markdown("#### Defina os fluxos permitidos entre setores")
        
        # Criar dataframe editável
        df_fluxos_input = pd.DataFrame(
            index=range(len(setores) * (len(setores) - 1)),
            columns=["De", "Para", "Capacidade", "Custo", "Juros (%)"]
        )
        
        # Preencher com fluxos existentes ou valores padrão
        idx = 0
        for i in setores:
            for j in setores:
                if i != j and i != 'A':
                    df_fluxos_input.loc[idx, "De"] = i
                    df_fluxos_input.loc[idx, "Para"] = j
                    df_fluxos_input.loc[idx, "Capacidade"] = 50000
                    df_fluxos_input.loc[idx, "Custo"] = 2.0
                    df_fluxos_input.loc[idx, "Juros (%)"] = 3.0
                    idx += 1
        
        # Remover linhas vazias
        df_fluxos_input = df_fluxos_input.dropna()
        
        # Editor de dados
        df_fluxos_editado = st.data_editor(
            df_fluxos_input,
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True,
            column_config={
                "De": st.column_config.SelectboxColumn("De", options=setores),
                "Para": st.column_config.SelectboxColumn("Para", options=setores),
                "Capacidade": st.column_config.NumberColumn(
                    "Capacidade",
                    min_value=0,
                    step=1000,
                    format="R$ %d"
                ),
                "Custo": st.column_config.NumberColumn(
                    "Custo",
                    min_value=0.0,
                    step=0.1,
                    format="%.2f"
                ),
                "Juros (%)": st.column_config.NumberColumn(
                    "Juros (%)",
                    min_value=0.0,
                    step=0.1,
                    format="%.2f"
                )
            }
        )
        
        # Atualizar fluxos
        fluxos = []
        for _, row in df_fluxos_editado.iterrows():
            if pd.notna(row["De"]) and pd.notna(row["Para"]):
                if row["De"] != row["Para"] and row["De"] != 'A':
                    fluxos.append((
                        row["De"],
                        row["Para"],
                        float(row["Capacidade"]),
                        float(row["Custo"]) * fator_ajuste,
                        float(row["Juros (%)"]) / 100 * fator_ajuste
                    ))

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

# Dashboard de visualização de dados
st.markdown("---")
st.markdown('<p class="section-header">Dados de Entrada</p>', unsafe_allow_html=True)

# Métricas resumidas
col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)

with col_metric1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Demanda Total</div>
        <div class="metric-value">R$ {:.0f}</div>
    </div>
    """.format(sum(max(0, demandas[(t, s)]) for t in periodos for s in setores)), unsafe_allow_html=True)

with col_metric2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Capacidade Total</div>
        <div class="metric-value">R$ {:.0f}</div>
    </div>
    """.format(sum(cap for (_, _, cap, _, _) in fluxos)), unsafe_allow_html=True)

with col_metric3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Fluxos Configurados</div>
        <div class="metric-value">{}</div>
    </div>
    """.format(len(fluxos)), unsafe_allow_html=True)

with col_metric4:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Cenário</div>
        <div class="metric-value">{}</div>
    </div>
    """.format(cenario), unsafe_allow_html=True)

# Visualização dos dados de entrada
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demandas por Período e Setor")
    df_demandas = pd.DataFrame([
        {'Período': t, 'Setor': s, 'Demanda': demandas[(t, s)]} 
        for (t, s) in demandas
    ])
    
    # Pivot table para melhor visualização
    pivot_demandas = df_demandas.pivot(index='Setor', columns='Período', values='Demanda')
    pivot_demandas.columns = [f'P{t}' for t in periodos]
    pivot_demandas['Total'] = pivot_demandas.sum(axis=1)
    
    st.dataframe(pivot_demandas, use_container_width=True)

with col2:
    st.subheader("Fluxos Permitidos")
    df_fluxos = pd.DataFrame(fluxos, columns=["De", "Para", "Capacidade", "Custo", "Juros"])
    df_fluxos['Custo'] = df_fluxos['Custo'].round(2)
    df_fluxos['Juros'] = (df_fluxos['Juros'] * 100).round(2)
    df_fluxos['Juros'] = df_fluxos['Juros'].astype(str) + '%'
    
    st.dataframe(df_fluxos, use_container_width=True)

# Botão de otimização
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
with col_btn2:
    botao_otimizar = st.button("Executar Otimização", type="primary", use_container_width=True)

if botao_otimizar:
    if not fluxos:
        st.error("Nenhum fluxo definido. Configure os fluxos permitidos antes de otimizar.")
    else:
        resultados = {}
        
        # Criar tabs para resultados
        tabs_resultados = st.tabs(["Sem Relaxamento", "Com Relaxamento", "Comparativo"])
        
        for idx, modo in enumerate(["Sem relaxamento", "Com relaxamento"]):
            with tabs_resultados[idx]:
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
                    
                    # Dashboard de métricas
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown("""
                        <div class="metric-card">
                            <div class="metric-label">Status</div>
                            <div class="metric-value">{}</div>
                        </div>
                        """.format(LpStatus[prob.status]), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div class="metric-card">
                            <div class="metric-label">Custo Total</div>
                            <div class="metric-value">R$ {:.0f}</div>
                        </div>
                        """.format(value(prob.objective)), unsafe_allow_html=True)
                    
                    with col3:
                        total_fluxo = sum(f[3] for f in fluxos_resultado)
                        st.markdown("""
                        <div class="metric-card">
                            <div class="metric-label">Volume Transacionado</div>
                            <div class="metric-value">R$ {:.0f}</div>
                        </div>
                        """.format(total_fluxo), unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown("""
                        <div class="metric-card">
                            <div class="metric-label">Fluxos Ativos</div>
                            <div class="metric-value">{}</div>
                        </div>
                        """.format(len(fluxos_resultado)), unsafe_allow_html=True)
                    
                    # Visualização do grafo direcionado
                    if fluxos_resultado:
                        st.markdown("### Rede de Fluxos")
                        df_fluxos_resultado = pd.DataFrame(fluxos_resultado, 
                                                           columns=["De", "Para", "Período", "Fluxo"])
                        
                        # Seletor de visualização
                        viz_tipo = st.radio(
                            "Tipo de Visualização",
                            ["Grafo Direcionado", "Diagrama de Sankey"],
                            horizontal=True,
                            key=f"viz_{modo}"
                        )
                        
                        if viz_tipo == "Grafo Direcionado":
                            fig_grafo = criar_grafo_direcionado_temporal(df_fluxos_resultado, modo)
                            if fig_grafo:
                                st.plotly_chart(fig_grafo, use_container_width=True)
                        else:
                            fig_sankey = criar_grafico_sankey(df_fluxos_resultado, modo)
                            if fig_sankey:
                                st.plotly_chart(fig_sankey, use_container_width=True)
                        
                        # Análise temporal
                        st.markdown("### Análise Temporal")
                        fig_comparativo = criar_grafico_comparativo(fluxos_resultado)
                        st.pyplot(fig_comparativo)
                        
                        # Tabela de fluxos detalhada
                        st.markdown("### Detalhamento dos Fluxos")
                        df_fluxos_resultado['Fluxo'] = df_fluxos_resultado['Fluxo'].round(2)
                        st.dataframe(df_fluxos_resultado, use_container_width=True)
                        
                        # Análise de utilização
                        st.markdown("### Utilização de Capacidade")
                        utilizacao_data = []
                        for (i, j, cap, _, _) in fluxos:
                            fluxo_total = df_fluxos_resultado[
                                (df_fluxos_resultado['De'] == i) & 
                                (df_fluxos_resultado['Para'] == j)
                            ]['Fluxo'].sum()
                            utilizacao = (fluxo_total / (cap * len(periodos))) * 100
                            utilizacao_data.append({
                                'De': i, 'Para': j,
                                'Fluxo Total': fluxo_total,
                                'Capacidade': cap,
                                'Utilização (%)': round(utilizacao, 1)
                            })
                        
                        df_utilizacao = pd.DataFrame(utilizacao_data)
                        df_utilizacao = df_utilizacao[df_utilizacao['Fluxo Total'] > 0]
                        
                        if not df_utilizacao.empty:
                            # Gráfico de barras horizontais
                            fig_utilizacao = px.bar(
                                df_utilizacao,
                                x='Utilização (%)',
                                y=[f"{row['De']}→{row['Para']}" for _, row in df_utilizacao.iterrows()],
                                orientation='h',
                                title='Utilização de Capacidade por Fluxo',
                                color='Utilização (%)',
                                color_continuous_scale='RdYlGn_r',
                                range_color=[0, 100]
                            )
                            fig_utilizacao.update_layout(
                                xaxis_title="Utilização (%)",
                                yaxis_title="Fluxo",
                                height=400
                            )
                            st.plotly_chart(fig_utilizacao, use_container_width=True)
                            
                            st.dataframe(df_utilizacao, use_container_width=True)
                    
                    # Exibir erros se houver
                    if erros_resultado:
                        st.markdown("### Demandas Não Atendidas")
                        df_erros = pd.DataFrame(erros_resultado, 
                                               columns=["Setor", "Período", "Erro", "Tipo"])
                        st.dataframe(df_erros, use_container_width=True)
        
        # Tab comparativo
        with tabs_resultados[2]:
            if len(resultados) == 2:
                st.markdown("### Comparação entre Modos")
                
                col1, col2 = st.columns(2)
                
                for idx, (modo, res) in enumerate(resultados.items()):
                    with (col1 if idx == 0 else col2):
                        st.markdown(f"#### {modo}")
                        st.markdown(f"**Status:** {res['status']}")
                        st.markdown(f"**Custo Total:** R$ {res['custo_total']:,.2f}")
                        st.markdown(f"**Número de Fluxos:** {len(res['fluxos'])}")
                        st.markdown(f"**Erros:** {len(res['erros'])}")
                
                # Gráfico comparativo
                fig_comparacao = go.Figure(data=[
                    go.Bar(
                        x=list(resultados.keys()),
                        y=[res['custo_total'] for res in resultados.values()],
                        text=[f"R$ {res['custo_total']:,.2f}" for res in resultados.values()],
                        textposition='auto',
                        marker_color=[COR_PRINCIPAL, COR_SECUNDARIA]
                    )
                ])
                
                fig_comparacao.update_layout(
                    title="Comparação de Custos Totais",
                    xaxis_title="Modo",
                    yaxis_title="Custo Total (R$)",
                    height=400
                )
                
                st.plotly_chart(fig_comparacao, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Desenvolvido para a disciplina MS529 - Otimização de Fluxo de Caixa</p>",
    unsafe_allow_html=True
)
