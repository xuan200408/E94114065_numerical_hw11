import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# === (a) Shooting Method ===
def shooting_method():
    from scipy.integrate import solve_ivp

    def ode_system(x, y):
        dy1 = y[1]
        dy2 = -(x + 1) * y[1] - 2 * y[0] + (1 - x**2) * np.exp(-x)
        return [dy1, dy2]

    def objective(s):
        sol = solve_ivp(ode_system, [0, 1], [1, s], t_eval=[1])
        return sol.y[0, -1] - 2  # y(1) = 2

    from scipy.optimize import root_scalar
    res = root_scalar(objective, bracket=[0, 5], method='brentq')
    s_star = res.root

    sol = solve_ivp(ode_system, [0, 1], [1, s_star], t_eval=np.linspace(0, 1, 101))
    x_vals, y_vals = sol.t, sol.y[0]

    return x_vals, y_vals

# === (b) Finite Difference Method ===
def finite_difference_method():
    h = 0.1
    x_vals = np.linspace(0, 1, 11)
    N = 9

    A = np.zeros((N, N))
    b = np.zeros(N)

    for i in range(N):
        xi = x_vals[i + 1]
        pi = -(xi + 1)
        qi = 2
        ri = (1 - xi**2) * np.exp(-xi)

        A[i, i] = 2 + h**2 * qi

        if i > 0:
            A[i, i - 1] = -1 - 0.5 * h * pi
        else:
            b[i] += (1 + 0.5 * h * pi) * 1

        if i < N - 1:
            A[i, i + 1] = -1 + 0.5 * h * pi
        else:
            b[i] += (1 - 0.5 * h * pi) * 2

        b[i] += h**2 * ri

    y_internal = np.linalg.solve(A, b)
    y_full = np.zeros(11)
    y_full[0] = 1
    y_full[1:-1] = y_internal
    y_full[-1] = 2

    return x_vals, y_full

# === (c) Variational Method ===
def variational_method():
    h = 0.1
    x_nodes = np.linspace(0, 1, 11)
    N = 9

    def y1(x):
        return 1 + x

    def phi(i, x):
        xi = x_nodes[i]
        left = np.where((x >= xi - h) & (x <= xi), (x - xi + h) / h, 0)
        right = np.where((x >= xi) & (x <= xi + h), (xi + h - x) / h, 0)
        return left + right

    def dphi(i, x):
        xi = x_nodes[i]
        return np.where((x >= xi - h) & (x <= xi), 1/h, 0) + \
               np.where((x >= xi) & (x <= xi + h), -1/h, 0)

    def inner_product(f, g):
        x_eval = np.linspace(0, 1, 101)
        return np.trapz(f(x_eval) * g(x_eval), x_eval)

    A = np.zeros((N, N))
    b = np.zeros(N)

    for i in range(1, N + 1):
        for j in range(1, N + 1):
            def integrand_A(x):
                return dphi(i, x) * dphi(j, x) + 2 * phi(i, x) * phi(j, x)
            A[i - 1, j - 1] = inner_product(lambda x: integrand_A(x), lambda x: 1)

        def rhs_func(x):
            f = 2 * y1(x) + (1 - x**2) * np.exp(-x)
            return f - 2 * y1(x)

        b[i - 1] = inner_product(rhs_func, lambda x: phi(i, x))

    c = np.linalg.solve(A, b)

    x_plot = np.linspace(0, 1, 101)
    y_vals = y1(x_plot)
    for i in range(1, N + 1):
        y_vals += c[i - 1] * phi(i, x_plot)

    y_nodes = y1(x_nodes)
    for i in range(1, N + 1):
        y_nodes[i] += c[i - 1]

    return x_nodes, y_nodes, x_plot, y_vals

# === 執行並顯示結果 ===
x_a, y_a = shooting_method()
x_b, y_b = finite_difference_method()
x_c, y_c_nodes, x_c_plot, y_c_plot = variational_method()

# === 輸出表格 ===
df = pd.DataFrame({
    'x': x_b,
    'Shooting': np.interp(x_b, x_a, y_a),
    'Finite Diff': y_b,
    'Variational': y_c_nodes
}).round(6)

print("\n=== 數值表 ===")
print(df.to_string(index=False))

