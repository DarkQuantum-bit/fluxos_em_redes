import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value

# Configurações iniciais
st.set_page_config(page_title="MS529 - Otimização de Fluxo de Caixa", layout="wide")

# CSS simplificado
st.markdown("""
<style>
    .main-header {
        background-color: #2c3e50;
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
        opacity: 0.9;
    }
    .section-title {
        color: #2c3e50;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        border-bottom: 2px solid #bdc3c7;
        padding-bottom: 0.5rem;
    }
    .stButton button {
        background-color: #2c3e50;
        color: white;
        font-weight: 500;
        border-radius: 4px;
        padding: 0.5rem 1.5rem;
    }
    .stButton button:hover {
        background-color: #34495e;
    }
</style>
""", unsafe_allow_html=True)

# Cores para gráficos
COR_PRINCIPAL = "#2c3e50"
COR_SECUNDARIA = "#3498db"
COR_TERCIARIA = "#e74c3c"
COR_QUATERNARIA = "#2ecc71"

# Definições de setores e períodos
setores = ['A', 'B', 'C', 'D', 'E', 'F']
periodos = [1, 2, 3]

# Estado da sessão
if 'mostrar_modelagem' not in st.session_state:
    st.session_state.mostrar_modelagem = False

def validar_dados(demandas, fluxos):
    """Valida se os dados inseridos são consistentes"""
    erros = []
    warnings = []
    
    # Verificar equilíbrio de demandas
    for t in periodos:
        total_positivo = sum(demandas[(t, s)] for s in setores if demandas[(t, s)] > 0)
        total_negativo = sum(demandas[(t, s)] for s in setores if demandas[(t, s)] < 0)
        if abs(total_positivo + total_negativo) > 0.01:
            erros.append(f"Período {t}: Demandas não equilibradas")
    
    # Verificar capacidades
    for (i, j, cap, custo, juros) in fluxos:
        if cap <= 0:
            erros.append(f"Fluxo {i}->{j}: Capacidade deve ser positiva")
        if custo < 0 or juros < 0:
            erros.append(f"Fluxo {i}->{j}: Custos e juros não podem ser negativos")
    
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
    
    # Custo de oportunidade do saldo (mantido simples)
    taxa_oportunidade = 0.01
    custo_oportunidade = lpSum(taxa_oportunidade * saldo[s, t] 
                               for s in setores 
                               for t in periodos)
    
    # Custo de juros sobre déficit
    taxa_juros_deficit = 0.03
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

def criar_grafico_comparativo(fluxos_resultado):
    """Cria gráfico de barras com volume por período"""
    if not fluxos_resultado:
        return None
    
    df = pd.DataFrame(fluxos_resultado, columns=["De", "Para", "Período", "Fluxo"])
    fluxos_por_periodo = df.groupby('Período')['Fluxo'].sum()
    
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(fluxos_por_periodo.index, fluxos_por_periodo.values, 
                  color=COR_SECUNDARIA, alpha=0.8)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'R$ {height:,.0f}',
                ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Período', fontsize=10)
    ax.set_ylabel('Volume de Transações (R$)', fontsize=10)
    ax.set_title('Volume de Transações por Período', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(periodos)
    
    plt.tight_layout()
    return fig

def criar_grafo_direcionado_temporal(df_fluxos, modo):
    """Cria grafo direcionado usando networkx e matplotlib"""
    if df_fluxos.empty:
        return None
    
    # Criar grafo direcionado
    G = nx.DiGraph()
    
    # Adicionar nós
    for s in setores:
        G.add_node(s)
    
    # Adicionar arestas com informações de fluxo
    for _, row in df_fluxos.iterrows():
        de, para, periodo, fluxo = row['De'], row['Para'], row['Período'], row['Fluxo']
        
        if G.has_edge(de, para):
            # Atualizar aresta existente
            G[de][para]['fluxo_total'] += fluxo
            G[de][para]['periodos'].append((periodo, fluxo))
        else:
            # Criar nova aresta
            G.add_edge(de, para, fluxo_total=fluxo, periodos=[(periodo, fluxo)])
    
    # Definir posições dos nós
    pos = {'A': (0, 0)}
    for idx, s in enumerate(['B', 'C', 'D', 'E', 'F']):
        angle = 2 * math.pi * idx / 5
        pos[s] = (5 * np.cos(angle), 5 * np.sin(angle))
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Desenhar nós
    nx.draw_networkx_nodes(G, pos, 
                          node_color=COR_PRINCIPAL, 
                          node_size=1500,
                          alpha=0.9,
                          ax=ax)
    
    # Desenhar labels dos nós
    nx.draw_networkx_labels(G, pos, 
                           font_size=14, 
                           font_weight='bold',
                           font_color='white',
                           ax=ax)
    
    # Desenhar arestas com espessura proporcional ao fluxo
    for (de, para, data) in G.edges(data=True):
        fluxo_total = data['fluxo_total']
        width = min(fluxo_total / 50000, 4)  # Normalizar espessura
        
        # Desenhar aresta
        nx.draw_networkx_edges(G, pos, 
                              edgelist=[(de, para)],
                              width=width,
                              edge_color=COR_SECUNDARIA,
                              alpha=0.7,
                              arrowsize=20,
                              connectionstyle='arc3, rad=0.1',
                              ax=ax)
        
        # Adicionar label com valor do fluxo
        x1, y1 = pos[de]
        x2, y2 = pos[para]
        x_medio = (x1 + x2) / 2
        y_medio = (y1 + y2) / 2
        
        # Formatar texto do label
        label = f"R$ {fluxo_total:,.0f}"
        
        ax.text(x_medio, y_medio, label, 
               fontsize=8, 
               ha='center', 
               va='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax.set_title(f'Grafo de Fluxos Direcionados - {modo}', fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

# Interface principal
st.markdown("""
<div class="main-header">
    <h1>MS529 - Otimização de Fluxo de Caixa</h1>
    <p>Projeto da disciplina de Fluxos em Redes</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Configurações")
    
    # Modo de dados
    modo_dados = st.radio(
        "Modo de Entrada de Dados",
        ["Gerar aleatoriamente", "Inserir manualmente"],
        key="modo_dados_radio"
    )
    
    st.divider()
    
    # Cenários
    st.subheader("Cenário de Análise")
    cenario = st.selectbox(
        "Selecione o cenário",
        ["Padrão", "Otimista", "Pessimista"],
        key="cenario_select"
    )
    
    # Fator de ajuste
    if cenario == "Otimista":
        fator_ajuste = 0.8
        st.caption("Custos 20% menores")
    elif cenario == "Pessimista":
        fator_ajuste = 1.2
        st.caption("Custos 20% maiores")
    else:
        fator_ajuste = 1.0
        st.caption("Custos normais")
    
    st.divider()
    
    # Parâmetros de otimização
    st.subheader("Parâmetros de Otimização")
    M = st.number_input(
        "Penalização por erro (M)",
        value=10.0,
        min_value=1.0,
        max_value=100.0,
        step=1.0,
        key="parametro_M"
    )
    
    st.divider()
    
    # Botão para mostrar modelagem
    if st.button("Mostrar Modelagem Matemática"):
        st.session_state.mostrar_modelagem = not st.session_state.mostrar_modelagem

# Área principal
if st.session_state.mostrar_modelagem:
    with st.expander("Modelagem Matemática", expanded=True):
        st.markdown(r"""
### Formulação Matemática

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
""")

# Inicialização das estruturas de dados
demandas = {}
fluxos = []

# Modo de geração aleatória
if modo_dados == "Gerar aleatoriamente":
    st.markdown("### Parâmetros de Geração de Dados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        seed = st.number_input("Seed aleatória", min_value=0, value=42, key="seed_input")
        demanda_total = st.number_input("Demanda total por período (R$)", value=400000, step=10000, key="demanda_total")
    
    with col2:
        cap_min = st.number_input("Capacidade mínima (R$)", value=30000, step=1000, key="cap_min")
        cap_max = st.number_input("Capacidade máxima (R$)", value=120000, step=1000, key="cap_max")
    
    with col3:
        custo_min = st.number_input("Custo unitário mínimo", value=1.0, step=0.1, key="custo_min")
        custo_max = st.number_input("Custo unitário máximo", value=3.0, step=0.1, key="custo_max")
    
    col4, col5 = st.columns(2)
    with col4:
        juros_min = st.number_input("Juros mínimo (%)", value=1.0, step=0.1, key="juros_min")
    with col5:
        juros_max = st.number_input("Juros máximo (%)", value=5.0, step=0.1, key="juros_max")
    
    np.random.seed(int(seed))
    
    # Gerar demandas
    for t in periodos:
        proporcoes = np.random.dirichlet(np.ones(len(setores) - 1), 1).flatten()
        for idx, s in enumerate([x for x in setores if x != 'A']):
            demandas[(t, s)] = int(demanda_total * proporcoes[idx])
        demandas[(t, 'A')] = -sum(demandas[(t, s)] for s in setores if s != 'A')
    
    # Gerar fluxos - CORRIGIDO: A pode enviar fluxos
    for i in setores:
        for j in setores:
            if i != j and j != 'A':  # A não recebe, mas pode enviar
                cap = np.random.randint(int(cap_min), int(cap_max))
                custo = np.round(np.random.uniform(custo_min, custo_max), 2)
                juros = np.round(np.random.uniform(juros_min / 100, juros_max / 100), 4)
                fluxos.append((i, j, cap, custo * fator_ajuste, juros * fator_ajuste))

# Modo de inserção manual
else:
    st.markdown("### Inserção Manual de Dados")
    
    tab_demandas, tab_fluxos = st.tabs(["Demandas", "Fluxos Permitidos"])
    
    with tab_demandas:
        st.markdown("#### Demandas por Setor e Período")
        
        df_demandas_input = pd.DataFrame(
            index=setores,
            columns=[f"Período {t}" for t in periodos]
        )
        
        for s in setores:
            for t in periodos:
                df_demandas_input.loc[s, f"Período {t}"] = 0
        
        df_demandas_editado = st.data_editor(
            df_demandas_input,
            use_container_width=True,
            num_rows="fixed",
            key="editor_demandas",
            column_config={
                **{f"Período {t}": st.column_config.NumberColumn(
                    f"Período {t}",
                    min_value=-1000000,
                    max_value=1000000,
                    step=1000,
                    format="%d"
                ) for t in periodos}
            }
        )
        
        for s in setores:
            for t in periodos:
                demandas[(t, s)] = float(df_demandas_editado.loc[s, f"Período {t}"])
        
        st.caption("Nota: Setor A deve ter demanda negativa (fornecedor) e demais setores positiva (consumidores).")
    
    with tab_fluxos:
        st.markdown("#### Fluxos Permitidos entre Setores")
        
        # Criar lista de fluxos padrão - CORRIGIDO: A pode enviar
        fluxos_padrao = []
        for i in setores:
            for j in setores:
                if i != j and j != 'A':  # A não recebe, mas pode enviar
                    fluxos_padrao.append({
                        "De": i,
                        "Para": j,
                        "Capacidade": 50000,
                        "Custo": 2.0,
                        "Juros (%)": 3.0
                    })
        
        df_fluxos_input = pd.DataFrame(fluxos_padrao)
        
        df_fluxos_editado = st.data_editor(
            df_fluxos_input,
            use_container_width=True,
            num_rows="dynamic",
            hide_index=True,
            key="editor_fluxos",
            column_config={
                "De": st.column_config.SelectboxColumn("De", options=setores),
                "Para": st.column_config.SelectboxColumn("Para", options=setores),
                "Capacidade": st.column_config.NumberColumn("Capacidade", min_value=0, step=1000, format="%d"),
                "Custo": st.column_config.NumberColumn("Custo", min_value=0.0, step=0.1, format="%.2f"),
                "Juros (%)": st.column_config.NumberColumn("Juros (%)", min_value=0.0, step=0.1, format="%.2f")
            }
        )
        
        fluxos = []
        for _, row in df_fluxos_editado.iterrows():
            if pd.notna(row["De"]) and pd.notna(row["Para"]):
                if row["De"] != row["Para"] and row["Para"] != 'A':
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

# Visualização dos dados de entrada
st.markdown("---")
st.markdown('<p class="section-title">Dados de Entrada</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demandas por Período e Setor")
    df_demandas = pd.DataFrame([
        {'Período': t, 'Setor': s, 'Demanda': demandas[(t, s)]} 
        for (t, s) in demandas
    ])
    
    pivot_demandas = df_demandas.pivot(index='Setor', columns='Período', values='Demanda')
    pivot_demandas.columns = [f'P{t}' for t in periodos]
    
    st.dataframe(pivot_demandas, use_container_width=True)

with col2:
    st.subheader("Fluxos Permitidos")
    df_fluxos = pd.DataFrame(fluxos, columns=["De", "Para", "Capacidade", "Custo", "Juros"])
    df_fluxos['Custo'] = df_fluxos['Custo'].round(2)
    df_fluxos['Juros'] = (df_fluxos['Juros'] * 100).round(2)
    
    st.dataframe(df_fluxos, use_container_width=True)

# Botão de otimização
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
with col_btn2:
    botao_otimizar = st.button("Executar Otimização", use_container_width=True)

if botao_otimizar:
    if not fluxos:
        st.error("Nenhum fluxo definido. Configure os fluxos permitidos antes de otimizar.")
    else:
        resultados = {}
        
        tabs_resultados = st.tabs(["Sem Relaxamento", "Com Relaxamento", "Comparativo"])
        
        for idx, modo in enumerate(["Sem relaxamento", "Com relaxamento"]):
            with tabs_resultados[idx]:
                with st.spinner(f"Resolvendo problema {modo}..."):
                    prob = criar_modelo_otimizacao(demandas, fluxos, modo, M)
                    
                    fluxos_resultado, erros_resultado, saldos_resultado = extrair_resultados(prob, modo)
                    
                    resultados[modo] = {
                        'status': LpStatus[prob.status],
                        'custo_total': value(prob.objective),
                        'fluxos': fluxos_resultado,
                        'erros': erros_resultado,
                        'saldos': saldos_resultado
                    }
                    
                    # Métricas principais
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Status", LpStatus[prob.status])
                    
                    with col2:
                        st.metric("Custo Total", f"R$ {value(prob.objective):,.2f}")
                    
                    with col3:
                        total_fluxo = sum(f[3] for f in fluxos_resultado)
                        st.metric("Volume Total", f"R$ {total_fluxo:,.2f}")
                    
                    # Visualização do grafo
                    if fluxos_resultado:
                        st.markdown("### Grafo de Fluxos")
                        df_fluxos_resultado = pd.DataFrame(fluxos_resultado, 
                                                           columns=["De", "Para", "Período", "Fluxo"])
                        
                        fig_grafo = criar_grafo_direcionado_temporal(df_fluxos_resultado, modo)
                        if fig_grafo:
                            st.pyplot(fig_grafo)
                        
                        # Análise temporal
                        st.markdown("### Análise Temporal")
                        fig_comparativo = criar_grafico_comparativo(fluxos_resultado)
                        if fig_comparativo:
                            st.pyplot(fig_comparativo)
                        
                        # Tabela de fluxos
                        st.markdown("### Detalhamento dos Fluxos")
                        df_fluxos_resultado['Fluxo'] = df_fluxos_resultado['Fluxo'].round(2)
                        st.dataframe(df_fluxos_resultado, use_container_width=True)
                    
                    # Exibir erros se houver
                    if erros_resultado:
                        st.markdown("### Demandas Não Atendidas")
                        df_erros = pd.DataFrame(erros_resultado, 
                                               columns=["Setor", "Período", "Erro", "Tipo"])
                        st.dataframe(df_erros, use_container_width=True)
                    
                    # Exibir saldos se houver
                    if saldos_resultado:
                        st.markdown("### Saldos por Setor e Período")
                        df_saldos = pd.DataFrame(saldos_resultado, 
                                                columns=["Setor", "Período", "Saldo"])
                        pivot_saldos = df_saldos.pivot(index='Setor', columns='Período', values='Saldo')
                        pivot_saldos.columns = [f'P{t}' for t in periodos]
                        st.dataframe(pivot_saldos, use_container_width=True)
        
        # Tab comparativo
        with tabs_resultados[2]:
            if len(resultados) == 2:
                st.markdown("### Comparação entre Modos")
                
                col1, col2 = st.columns(2)
                
                for idx, (modo, res) in enumerate(resultados.items()):
                    with (col1 if idx == 0 else col2):
                        st.markdown(f"**{modo}**")
                        st.markdown(f"Status: {res['status']}")
                        st.markdown(f"Custo Total: R$ {res['custo_total']:,.2f}")
                        st.markdown(f"Fluxos: {len(res['fluxos'])}")
                        st.markdown(f"Erros: {len(res['erros'])}")
                
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
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig_comparacao, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666;'>Projeto desenvolvido para a disciplina MS529 - Fluxos em Redes</p>",
    unsafe_allow_html=True
)
