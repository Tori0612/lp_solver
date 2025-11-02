import streamlit as st
import pulp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

# --- CONFIGURAÇÃO DA PÁGINA ---
def to_subscript(n: int):
    subs = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
    return str(n).translate(subs)

st.set_page_config(layout="wide")

# --- NOVO: LÓGICA DE SENHA COM SESSION STATE ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

try:
    PASSWORD = st.secrets['auth']['password']
except (FileNotFoundError, KeyError):
    st.error("Arquivo de 'secrets' não encontrado. Defina uma senha padrão se estiver rodando localmente.")
    PASSWORD = "admin" # Senha fallback

if not st.session_state.authenticated:
    st.title('Login Necessário')
    password = st.text_input('Digite a senha para acessar o app:', type='password')
    
    if password == PASSWORD:
        st.session_state.authenticated = True
        st.rerun() # Re-executa o script para mostrar o app principal
    elif password != "":
        st.warning('Acesso restrito. Digite a senha correta para continuar.')
    
    # Para a execução se não estiver logado
    st.stop() 

# --- APLICAÇÃO PRINCIPAL (Só executa se st.session_state.authenticated == True) ---

st.title("Solver Avançado de PL com Análise de Sensibilidade")

# --- NOVO: SESSION STATE PARA O BOTÃO RESOLVER ---
if 'solution_run' not in st.session_state:
    st.session_state.solution_run = False

# --- 1. PROBLEM DEFINITION ---
st.header("1. Definição do Problema")

col1, col2, col3 = st.columns(3)
with col1:
    problem_type = st.radio("Tipo de Problema", ["Maximizar", "Minimizar"])
    
with col2:
    num_vars = st.number_input("Número de Variáveis ($x_1$, $x_2$, ...)", min_value=1, value=2, step=1)

with col3:
    num_constraints = st.number_input("Número de Restrições", min_value=1, value=1, step=1)

# Initialize lists to store coefficients
obj_coeffs = []
const_matrix = []
const_ops = []
const_rhs = []

# --- 2. OBJECTIVE FUNCTION INPUTS ---
st.header("2. Função Objetivo")
st.write(f"Defina os {num_vars} coeficientes da função objetivo. ex: $f(x) = C_1x_1 + C_2x_2 + ... + C_nx_n$")

obj_cols = st.columns(num_vars)
for i in range(num_vars):
    with obj_cols[i]:
        c = st.number_input(f"$C_{i+1}$ (para $x_{i+1}$)", key=f"c_{i}", value=1.0)
        obj_coeffs.append(c)

# --- 3. CONSTRAINTS INPUTS ---
st.header("3. Restrições")
st.write("Defina os coeficientes, operador e o Lado Direito (RHS) para cada restrição. ex: $A_1x_1 + A_2x_2 \leq B_1$")

for i in range(num_constraints):
    st.subheader(f"Restrição {i+1}")
    
    const_cols = st.columns(num_vars + 2)
    row_coeffs = []
    
    for j in range(num_vars):
        with const_cols[j]:
            a = st.number_input(f"$A_{j+1}$ (para $x_{j+1}$)", key=f"a_{i}_{j}", value=1.0)
            row_coeffs.append(a)
    
    with const_cols[num_vars]:
        op = st.selectbox(f"op({i+1})", ["<=", ">=", "=="], key=f"op_{i}")
        
    with const_cols[num_vars + 1]:
        b = st.number_input(f"$B_{i+1}$ (RHS)", key=f"b_{i}", value=10.0)

    const_matrix.append(row_coeffs)
    const_ops.append(op)
    const_rhs.append(b)


# --- 4. SOLVE BUTTON AND LOGIC ---
st.header("4. Resolver")

# NOVO: O botão agora *apenas* define o 'flag'
if st.button("Resolver", type="primary"):
    st.session_state.solution_run = True

# --- 5. DISPLAY RESULTS (HANDLING ALL CASES) ---
# NOVO: Este bloco inteiro agora depende do 'flag' do session_state
if st.session_state.solution_run:
    
    # --- A LÓGICA DE SOLVER FOI MOVIDA PARA AQUI DENTRO ---
    if problem_type == "Maximizar":
        prob = pulp.LpProblem("My_LP_Problem", pulp.LpMaximize)
    else:
        prob = pulp.LpProblem("My_LP_Problem", pulp.LpMinimize)

    variables = [pulp.LpVariable(f'x{to_subscript(i+1)}', lowBound=0) for i in range(num_vars)]
    prob += pulp.lpSum([obj_coeffs[i] * variables[i] for i in range(num_vars)]), "Objective_Function"

    for i in range(num_constraints):
        expr = pulp.lpSum([const_matrix[i][j] * variables[j] for j in range(num_vars)])
        op = const_ops[i]
        rhs = const_rhs[i]
        
        if op == "<=":
            prob += expr <= rhs, f"R{to_subscript(i+1)}"
        elif op == ">=":
            prob += expr >= rhs, f"R{to_subscript(i+1)}"
        elif op == "==":
            prob += expr == rhs, f"R{to_subscript(i+1)}"

    prob.solve()

    # --- TODO O RESTANTE DA LÓGICA DE DISPLAY FICA AQUI DENTRO ---
    st.header("Relatório da Solução")
    status = pulp.LpStatus[prob.status]
    st.write(f"**Status:** {status}")

    if prob.status == pulp.LpStatusOptimal:
        st.success("Solução Ótima Encontrada!")
        
        st.subheader("Valor da Função Objetivo")
        original_objective: float = pulp.value(prob.objective)
        st.metric("Total", f"{original_objective:.2f}")

        # --- Variables Report ---
        st.subheader("Variáveis de Decisão")
        var_data = []
        for v in prob.variables():
            var_data.append({
                'Variável': v.name,
                'Valor': v.varValue or 0.0,
                'Custo Reduzido': v.dj or 0.0
            })
        var_df = pd.DataFrame(var_data).set_index('Variável')
        st.dataframe(var_df)

        # --- Sensitivity Analysis Report ---
        st.subheader("Análise de Sensibilidade (Restrições)")
        const_data = []
        original_shadow_prices = {} 
        
        for i in range(num_constraints):
            name = f"R{to_subscript(i+1)}"
            constraint = prob.constraints[name]
            
            op = const_ops[i]
            rhs = const_rhs[i]
            slack = constraint.slack or 0.0
            shadow_price = constraint.pi or 0.0
            
            original_shadow_prices[name] = shadow_price
            
            final_value = 0
            if op == "<=":
                final_value = rhs - slack
            elif op == ">=":
                final_value = rhs + slack 
            elif op == "==":
                final_value = rhs
                
            const_data.append({
                'Restrição': name,
                'Valor Final (LHS)': final_value,
                'Operador': op,
                'Limite (RHS)': rhs,
                'Folga (Slack)': slack,
                'Preço Sombra (Dual)': shadow_price
            })

        col_order = ['Valor Final (LHS)', 'Operador', 'Limite (RHS)', 'Folga (Slack)', 'Preço Sombra (Dual)']
        const_df = pd.DataFrame(const_data).set_index('Restrição')
        const_df = const_df[col_order]
        st.dataframe(const_df)
        
        # --- 6. GRAPHICAL SOLUTION PLOT (2D ONLY) ---
        st.subheader("Solução Gráfica (Apenas problemas 2D)")
        
        if num_vars == 2:
            try:
                opt_x1 = variables[0].varValue or 0.0
                opt_x2 = variables[1].varValue or 0.0

                max_coord_val = max(opt_x1, opt_x2, 1)
                plot_limit = max_coord_val * 1.25 + 10
                
                x_vals = np.linspace(0, plot_limit, 400)
                fig, ax = plt.subplots(figsize=(6, 4.5))
                
                for i in range(num_constraints):
                    a1 = const_matrix[i][0]
                    a2 = const_matrix[i][1]
                    b = const_rhs[i]
                    
                    if a1 == 0 and a2 == 0: continue
                    elif a2 == 0 and a1 != 0:
                        x_line = b / a1
                        ax.axvline(x=x_line, label=f"$R_{i+1}: {a1}x_1 = {b}$")
                    elif a1 == 0 and a2 != 0:
                        y_line = b / a2
                        ax.axhline(y=y_line, label=f"$R_{i+1}: {a2}x_2 = {b}$")
                    elif a1 != 0 and a2 != 0:
                        y_vals = (b - a1 * x_vals) / a2
                        ax.plot(x_vals, y_vals, label=f"$R_{i+1}: {a1}x_1 + {a2}x_2 = {b}$")

                ax.plot(opt_x1, opt_x2, 'ro', markersize=10, 
                        label=f'Ótimo: ({opt_x1:.2f}, {opt_x2:.2f})')
                
                y_fills = []
                for i in range(num_constraints):
                    if const_ops[i] == "<=" and const_matrix[i][1] != 0:
                        y_vals = (const_rhs[i] - const_matrix[i][0] * x_vals) / const_matrix[i][1]
                        y_fills.append(y_vals)
                
                if y_fills:
                    y_min = np.minimum.reduce(y_fills)
                    ax.fill_between(x_vals, 0, y_min, where=(y_min >= 0), 
                                    color='grey', alpha=0.3, label='Região Viável (Est.)')

                ax.set_xlabel("$x_1$")
                ax.set_ylabel("$x_2$")
                ax.set_xlim(0, plot_limit)
                ax.set_ylim(0, plot_limit)
                ax.legend(fontsize="small")
                ax.grid(True)
                st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Ocorreu um erro ao tentar plotar o gráfico: {e}")
        else:
            st.info("O gráfico da solução está disponível apenas para problemas com 2 variáveis.")

        # --- 7. "WHAT IF" ANALYSIS SECTION ---
        # (Isto agora funciona, pois está dentro do bloco 'if st.session_state.solution_run:')
        st.subheader("Análise Interativa 'E Se?' (What-If)")
        st.write("Explore o impacto de mudar o RHS (Limite) de uma restrição.")

        constraint_names = const_df.index.tolist()
        
        selected_const_name = st.selectbox("Escolha uma restrição para analisar:", 
                                           options=constraint_names)
        
        if selected_const_name: 
            selected_index = constraint_names.index(selected_const_name)
            original_rhs = const_rhs[selected_index]

            range_delta = max(10.0, abs(original_rhs * 0.5))
            min_val = max(0.0, original_rhs - range_delta) 
            max_val = original_rhs + 1.5 * range_delta
            
            new_rhs = st.slider("Selecione o Novo Valor para o Limite (RHS):", 
                                min_value=min_val, 
                                max_value=max_val, 
                                value=original_rhs)

            if new_rhs != original_rhs:
                temp_rhs = list(const_rhs)
                temp_rhs[selected_index] = new_rhs
                
                if problem_type == "Maximizar":
                    prob_whatif = pulp.LpProblem("WhatIf_Problem", pulp.LpMaximize)
                else:
                    prob_whatif = pulp.LpProblem("WhatIf_Problem", pulp.LpMinimize)
                
                variables_whatif = [pulp.LpVariable(f'x{to_subscript(i+1)}', lowBound=0) for i in range(num_vars)]
                prob_whatif += pulp.lpSum([obj_coeffs[i] * variables_whatif[i] for i in range(num_vars)]), "Objective_Function"

                for i in range(num_constraints):
                    expr = pulp.lpSum([const_matrix[i][j] * variables_whatif[j] for j in range(num_vars)])
                    op = const_ops[i]
                    rhs_val = temp_rhs[i]
                    name = f"R{to_subscript(i+1)}" 
                    
                    if op == "<=": prob_whatif += expr <= rhs_val, name
                    elif op == ">=": prob_whatif += expr >= rhs_val, name
                    elif op == "==": prob_whatif += expr == rhs_val, name
                
                prob_whatif.solve()
                
                st.markdown(f"#### Resultados para {selected_const_name} com RHS = {new_rhs:.2f}")
                if prob_whatif.status == pulp.LpStatusOptimal:
                    new_obj: float = pulp.value(prob_whatif.objective)
                    
                    # Correção: Certifique-se de que os nomes das chaves são strings
                    new_shadow = prob_whatif.constraints[str(selected_const_name)].pi
                    original_shadow = original_shadow_prices[str(selected_const_name)]
                    
                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric("Novo Valor Objetivo", 
                                  f"{new_obj:.2f}", 
                                  delta=f"{(new_obj - original_objective):.2f}")
                    with res_col2:
                        st.metric(f"Novo Preço Sombra ({selected_const_name})", 
                                  f"{new_shadow:.2f}", 
                                  delta=f"{(new_shadow - (original_shadow or 0.0)):.2f}") # Adicionado 'or 0.0'
                else:
                    st.error(f"Com RHS = {new_rhs:.2f}, o problema se tornou: **{pulp.LpStatus[prob_whatif.status]}**")
            else:
                st.info("Mova o slider para analisar o impacto de um novo valor de RHS.")
        # --- 8. BOTÃO DE DOWNLOAD (VERSÃO EXCEL) ---
        st.subheader("8. Salvar Relatório")

        # --- NOVO: Função para criar o arquivo Excel em memória ---
        def to_excel():
            output = BytesIO()
            # Inicia o "escritor" de Excel
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                
                # --- Aba 1: Sumário do Problema ---
                # Cria um DataFrame simples para o sumário
                sumario_data = {
                    "Parâmetro": ["Tipo de Problema", "Função Objetivo"],
                    "Valor": [
                        problem_type, 
                        " + ".join([f"{obj_coeffs[i]}*x{to_subscript(i+1)}" for i in range(num_vars)])
                        ]
                }
                df_sumario = pd.DataFrame(sumario_data)
                df_sumario.to_excel(writer, sheet_name='Sumário', index=False)
                
                # --- Aba 2: Variáveis de Decisão ---
                # (var_df já existe)
                var_df.to_excel(writer, sheet_name='Variáveis de Decisão')
                
                # --- Aba 3: Análise de Sensibilidade ---
                # (const_df já existe)
                const_df.to_excel(writer, sheet_name='Análise de Restrições')
            
            # Pega os dados do arquivo em memória
            processed_data = output.getvalue()
            return processed_data
        # --- Fim da função ---

        # Cria o arquivo excel
        excel_data = to_excel()

        # Cria o botão de download
        st.download_button(
            label="Baixar Relatório da Solução (.xlsx)",
            data=excel_data,
            file_name=f"solucao_lp_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
            # MIME type para arquivos Excel
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" 
        )

    else:
        st.error(f"O solver não encontrou uma solução ótima. Status: {status}")
        st.write("Isso pode ser porque o problema é **Infactivel** (nenhuma solução satisfaz todas as restrições) ou **Ilimitado** (o objetivo pode ir ao infinito).")
        st.session_state.solution_run = False # Reseta se a solução falhar

else:
    # Mensagem inicial antes de 'Resolver' ser clicado
    st.info("Preencha os dados e clique em 'Resolver' para ver a solução.")
