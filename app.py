import streamlit as st
import pulp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- PASSWORD PROTECTION ---
PASSWORD = st.secrets['auth']['password']

st.set_page_config(layout="wide")

st.title('Login Necessário')
password = st.text_input('Digite a senha para acessar o app:', type='password')
st.title("Solver Avançado de PL com Análise de Sensibilidade")

if password != PASSWORD:
    st.warning('Acesso restrito papai, digite a senha correta para continuar.')
    st.stop()

# --- 1. PROBLEM DEFINITION ---
st.header("1. Definição do Problema")

col1, col2, col3 = st.columns(3)
with col1:
    # Requirement 1: Choose Maximize or Minimize
    problem_type = st.radio("Tipo de Problema", ["Maximizar", "Minimizar"])
    
with col2:
    # Requirement 2: Define number of variables
    num_vars = st.number_input("Número de Variáveis ($x_1$, $x_2$, ...)", min_value=1, value=2, step=1)

with col3:
    # Requirement 2: Define number of constraints
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
        # Use a unique key for each widget
        c = st.number_input(f"$C_{i+1}$ (para $x_{i+1}$)", key=f"c_{i}", value=1.0)
        obj_coeffs.append(c)

# --- 3. CONSTRAINTS INPUTS ---
st.header("3. Restrições")
st.write("Defina os coeficientes, operador e o Lado Direito (RHS) para cada restrição. ex: $A_1x_1 + A_2x_2 \\leq B_1$")

for i in range(num_constraints):
    st.subheader(f"Restrição {i+1}")
    
    # +2 columns for the operator (<=, >=, ==) and the RHS value
    const_cols = st.columns(num_vars + 2)
    row_coeffs = []
    
    # Get coefficients for each variable in this constraint
    for j in range(num_vars):
        with const_cols[j]:
            a = st.number_input(f"$A_{j+1}$ (para $x_{j+1}$)", key=f"a_{i}_{j}", value=1.0)
            row_coeffs.append(a)
    
    # Get the operator
    with const_cols[num_vars]:
        op = st.selectbox(f"op({i+1})", ["<=", ">=", "=="], key=f"op_{i}")
        
    # Get the RHS value
    with const_cols[num_vars + 1]:
        b = st.number_input(f"$B_{i+1}$ (RHS)", key=f"b_{i}", value=10.0)

    # Store all constraint data
    const_matrix.append(row_coeffs)
    const_ops.append(op)
    const_rhs.append(b)


# --- 4. SOLVE BUTTON AND LOGIC ---
st.header("4. Resolver")
if st.button("Resolver", type="primary"):
    
    # --- Setup the PuLP Problem ---
    if problem_type == "Maximizar":
        prob = pulp.LpProblem("My_LP_Problem", pulp.LpMaximize)
    else:
        prob = pulp.LpProblem("My_LP_Problem", pulp.LpMinimize)

    # Define variables
    variables = [pulp.LpVariable(f'x{i+1}', lowBound=0) for i in range(num_vars)]

    # Add objective function
    prob += pulp.lpSum([obj_coeffs[i] * variables[i] for i in range(num_vars)]), "Objective_Function"

    # Add constraints
    for i in range(num_constraints):
        expr = pulp.lpSum([const_matrix[i][j] * variables[j] for j in range(num_vars)])
        op = const_ops[i]
        rhs = const_rhs[i]
        
        if op == "<=":
            prob += expr <= rhs, f"Restricao_{i+1}"
        elif op == ">=":
            prob += expr >= rhs, f"Restricao_{i+1}"
        elif op == "==":
            prob += expr == rhs, f"Restricao_{i+1}"

    # --- Solve the problem ---
    prob.solve()

    # --- 5. DISPLAY RESULTS (HANDLING ALL CASES) ---
    st.header("Relatório da Solução")
    
    # Get the status as a string
    status = pulp.LpStatus[prob.status]
    st.write(f"**Status:** {status}")

    # Requirement 3: Handle exceptions / non-optimal solutions
    if prob.status == pulp.LpStatusOptimal:
        st.success("Solução Ótima Encontrada!")
        
        st.subheader("Valor da Função Objetivo")
        st.metric("Total", f"{pulp.value(prob.objective):.2f}")

        # Requirement 4: Structured Reports using Pandas DataFrames
        
        # --- Variables Report ---
        st.subheader("Variáveis de Decisão")
        var_data = []
        for v in prob.variables():
            var_data.append({
                'Variável': v.name,
                'Valor': v.varValue,
                'Custo Reduzido': v.dj
            })
        var_df = pd.DataFrame(var_data).set_index('Variável')
        st.dataframe(var_df)

        # --- Sensitivity Analysis Report ---
        st.subheader("Análise de Sensibilidade (Restrições)")
        const_data = []
        for name, c in prob.constraints.items():
            const_data.append({
                'Restrição': name,
                'Preço Sombra (Dual)': c.pi,
                'Folga (RHS - Valor)': c.slack
            })
        const_df = pd.DataFrame(const_data).set_index('Restrição')
        st.dataframe(const_df)
        
        # --- 6. GRAPHICAL SOLUTION PLOT (2D ONLY) ---
        st.subheader("Solução Gráfica (Apenas problemas 2D)")
        
        # We can only plot if we have exactly 2 variables
        if num_vars == 2:
            try:
                # Get the optimal values
                opt_x1 = variables[0].varValue
                opt_x2 = variables[1].varValue

                # Determine a good plot range, add 25% buffer
                max_coord_val = max(opt_x1, opt_x2, 1) # type: ignore
                plot_limit = max_coord_val * 1.25 + 10 # Add 10 for small values
                
                # Create a 400-point range for x1
                x_vals = np.linspace(0, plot_limit, 400)
                
                # Create the plot with a specific size
                fig, ax = plt.subplots(figsize=(6, 4.5))
                
                # Plot the constraints
                for i in range(num_constraints):
                    a1 = const_matrix[i][0]
                    a2 = const_matrix[i][1]
                    b = const_rhs[i]
                    
                    # Skip 0*x1 + 0*x2 <= b...
                    if a1 == 0 and a2 == 0:
                        continue
                    
                    # Handle vertical line (a2 = 0)
                    elif a2 == 0:
                        x_line = b / a1
                        ax.axvline(x=x_line, label=f"R{i+1}: {a1}x1 = {b}")
                    
                    # Handle horizontal line (a1 = 0)
                    elif a1 == 0:
                        y_line = b / a2
                        ax.axhline(y=y_line, label=f"R{i+1}: {a2}x2 = {b}")
                    
                    # Handle diagonal lines
                    else:
                        # y = (b - a1*x) / a2
                        y_vals = (b - a1 * x_vals) / a2
                        ax.plot(x_vals, y_vals, label=f"R{i+1}: {a1}x1 + {a2}x2 = {b}")

                # --- Mark the Optimal Point (Your Request!) ---
                ax.plot(opt_x1, opt_x2, 'ro', markersize=10, #type: ignore
                        label=f'Ótimo: ({opt_x1:.2f}, {opt_x2:.2f})')
                
                # --- Plot the Feasible Region (Simple) ---
                y_fills = []
                for i in range(num_constraints):
                    # Only consider non-horizontal lines for filling
                    if const_ops[i] == "<=" and const_matrix[i][1] != 0: # a2 != 0
                        y_vals = (const_rhs[i] - const_matrix[i][0] * x_vals) / const_matrix[i][1]
                        y_fills.append(y_vals)
                
                if y_fills:
                    # Find the minimum of all "less than" constraint lines
                    y_min = np.minimum.reduce(y_fills)
                    # Fill between the x-axis and the lowest line
                    ax.fill_between(x_vals, 0, y_min, where=(y_min >= 0), 
                                    color='grey', alpha=0.3, label='Região Viável (Est.)')

                # --- Final Plot Formatting ---
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                ax.set_xlim(0, plot_limit)
                ax.set_ylim(0, plot_limit)
                ax.legend(fontsize="small")
                ax.grid(True)
                
                st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Ocorreu um erro ao tentar plotar o gráfico: {e}")

        else:
            st.info("O gráfico da solução está disponível apenas para problemas com 2 variáveis.")

    else:
        # Handle Infeasible, Unbounded, etc.
        st.error(f"O solver não encontrou uma solução ótima. Status: {status}")
        st.write("Isso pode ser porque o problema é **Infactivel** (nenhuma solução satisfaz todas as restrições) ou **Ilimitado** (o objetivo pode ir ao infinito).")
