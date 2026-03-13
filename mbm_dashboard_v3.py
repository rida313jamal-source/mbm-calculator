import math
import numpy as np
import streamlit as st
import sympy as sp

# ============================================================
# MBM Research Dashboard (V3)
# Based on Version II structure
# Author: Rida Jamal Badawi Abu Sokon
# ============================================================

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="MBM Research Dashboard",
    page_icon="∫",
    layout="wide",
)

# ---------------------------
# Styling
# ---------------------------
st.markdown(
    """
    <style>
    .block-container { padding-top: 1.15rem; padding-bottom: 2.2rem; }
    .katex-display { margin: 0.8rem 0 0.8rem 0 !important; }
    h1, h2, h3 { letter-spacing: 0.2px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# Helper functions
# ============================================================
def latex_matrix(A, precision=6):
    A = np.array(A, dtype=float)
    rows = []
    for r in A:
        rows.append(" & ".join([f"{v:.{precision}g}" for v in r]))
    body = r"\\ ".join(rows)
    return r"\begin{bmatrix}" + body + r"\end{bmatrix}"

def latex_vector(v, precision=8):
    v = np.array(v, dtype=float).reshape(-1)
    body = r"\\ ".join([f"{x:.{precision}g}" for x in v])
    return r"\begin{bmatrix}" + body + r"\end{bmatrix}"

def scalar_from_mat(M):
    return float(np.array(M, dtype=float).reshape(-1)[0])

# ============================================================
# Basis builders
# ============================================================
def build_poly_basis_from_expr(poly_expr):
    x = sp.Symbol("x")
    P = sp.Poly(sp.expand(poly_expr), x)
    deg = int(P.degree())
    coeffs = [float(P.nth(i)) for i in range(deg + 1)]

    def Phi(t):
        return np.array([t**i for i in range(deg + 1)], dtype=float).reshape(-1, 1)

    m = deg + 1
    Omega = np.zeros((m, m), dtype=float)
    for i in range(1, m):
        Omega[i - 1, i] = i

    u = np.array(coeffs, dtype=float).reshape(-1, 1)
    return Phi, Omega, u, deg

def build_trig_basis(u0, v0, w):
    def Phi(t):
        return np.array([math.cos(w * t), math.sin(w * t)], dtype=float).reshape(-1, 1)

    Omega = w * np.array([[0, -1], [1, 0]], dtype=float)
    u = np.array([u0, v0], dtype=float).reshape(-1, 1)
    return Phi, Omega, u

def build_exp_trig_basis(alpha, u0, v0, w):
    def Phi(t):
        ex = math.exp(alpha * t)
        return np.array([ex * math.cos(w * t), ex * math.sin(w * t)], dtype=float).reshape(-1, 1)

    Omega = np.array([[alpha, -w], [w, alpha]], dtype=float)
    u = np.array([u0, v0], dtype=float).reshape(-1, 1)
    return Phi, Omega, u

def build_xcos_basis(w):
    # Phi = [cos(wx), sin(wx), x cos(wx), x sin(wx)]^T
    def Phi(t):
        c = math.cos(w * t)
        s = math.sin(w * t)
        return np.array([c, s, t * c, t * s], dtype=float).reshape(-1, 1)

    Omega = np.array(
        [
            [0, -w, 0, 0],
            [w, 0, 0, 0],
            [1, 0, 0, -w],
            [0, 1, w, 0],
        ],
        dtype=float,
    )
    u = np.array([0, 0, 1, 0], dtype=float).reshape(-1, 1)
    return Phi, Omega, u

def build_x2cos_basis(w):
    # Phi = [cos, sin, x cos, x sin, x^2 cos, x^2 sin]^T
    def Phi(t):
        c = math.cos(w * t)
        s = math.sin(w * t)
        return np.array(
            [c, s, t * c, t * s, (t**2) * c, (t**2) * s],
            dtype=float
        ).reshape(-1, 1)

    Omega = np.array(
        [
            [0, -w, 0, 0, 0, 0],
            [w, 0, 0, 0, 0, 0],
            [1, 0, 0, -w, 0, 0],
            [0, 1, w, 0, 0, 0],
            [0, 0, 2, 0, 0, -w],
            [0, 0, 0, 2, w, 0],
        ],
        dtype=float,
    )
    u = np.array([0, 0, 0, 0, 1, 0], dtype=float).reshape(-1, 1)
    return Phi, Omega, u

# ============================================================
# MBM engines
# ============================================================
def mbm_integral(Phi, Omega, u, a, b):
    m = Omega.shape[0]
    A = b * np.eye(m) + Omega.T
    z = np.linalg.solve(A, u)

    def F(t):
        return math.exp(b * t) * scalar_from_mat(Phi(t).T @ z)

    value = F(a) - F(0.0)
    return value, A, z, F

def mbm_laplace_truncated(Phi, Omega, u, s, a):
    # ∫_0^a e^{-s x} f(x) dx = MBM integral with b = -s
    return mbm_integral(Phi, Omega, u, a, -s)

def matrix_laplace_limit(Omega, u, Phi0, s):
    # L{f}(s) = u^T (sI - Omega)^(-1) Phi(0)
    m = Omega.shape[0]
    A = s * np.eye(m) - Omega
    val = scalar_from_mat(u.T @ np.linalg.solve(A, Phi0))
    return val, A

def mbm_first_order_ode(Phi, Omega, u, c, y0, x_eval):
    # y' - c y = f(x), y(0)=y0
    inner_val, A, z, F = mbm_integral(Phi, Omega, u, x_eval, -c)
    y_val = math.exp(c * x_eval) * (y0 + inner_val)
    return y_val, inner_val, A, z, F

# ============================================================
# Polynomial diagonal-Laplace rule
# ============================================================
def ascending_factorial(i, k):
    out = 1
    for j in range(k):
        out *= (i + j)
    return out

def polynomial_diagonal_laplace_matrix(n, t):
    # Indexed mathematically from 1..n+1
    M = np.zeros((n + 1, n + 1), dtype=float)
    for i in range(n + 1):
        for j in range(i, n + 1):
            k = j - i
            i_math = i + 1
            val = ((-1)**k) * ascending_factorial(i_math, k) * math.factorial(k) / (t ** (k + 1))
            M[i, j] = val
    return M

# ============================================================
# Header
# ============================================================
st.title("Matrix Boundary Method: Truncated Laplace Integrals and Differential Equations")
st.caption("Version II")
st.caption("By: Rida Jamal Badawi Abu Sokon")

# ============================================================
# Sidebar
# ============================================================
section = st.sidebar.radio(
    "Go to",
    [
        "Introduction",
        "Linear Spaces of Functions",
        "Finite-Dimensional Closure and the MBM Representation",
        "Operator Interpretation and Basic Laws",
        "Canonical Bases and Ready Matrices",
        "Worked Examples for Integrals",
        "Generalization: MBM for Products",
        "Exponential-Matrix Variant",
        "Laplace Connection",
        "Part II: Linear ODEs via MBM",
        "Integral Calculator",
        "Laplace Calculator",
        "ODE Calculator",
    ]
)

# ============================================================
# 1) Introduction
# ============================================================
if section == "Introduction":
    st.header("1. Introduction")

    st.markdown(
        """
        The Matrix Boundary Method (MBM) is a finite-dimensional algebraic framework
        for evaluating structured integrals and solving linear differential equations.
        The essential idea is to encode a function into a finite-dimensional state
        vector whose derivative is governed by a constant matrix. Once this state-space
        representation is available, integrals can be converted into matrix problems,
        and the final result becomes a boundary evaluation rather than a direct
        symbolic integration process.
        """
    )

    st.latex(r"\Phi'(x)=\Omega\Phi(x),\qquad f(x)=u^{\mathsf T}\Phi(x)")
    st.latex(r"\int_0^a e^{bx}f(x)\,dx=\left[e^{bx}\Phi(x)^{\mathsf T}(bI+\Omega^{\mathsf T})^{-1}u\right]_0^a")

    st.markdown(
        """
        The method is especially effective when the chosen function space is:
        - finite-dimensional,
        - closed under differentiation,
        - described by a basis with a constant derivative matrix.

        This dashboard follows the same structure as the research manuscript and
        supplements it with interactive matrix-based calculators.
        """
    )

# ============================================================
# 2) Linear Spaces of Functions
# ============================================================
elif section == "Linear Spaces of Functions":
    st.header("2. Linear Spaces of Functions: Concepts and Structure")

    st.subheader("2.1 The meaning of space")
    st.markdown(
        """
        In mathematics, a space is a collection of objects equipped with a clear
        internal structure. The emphasis is not on the kind of objects alone, but
        on the allowed operations and the rules that keep the collection stable.
        """
    )

    st.subheader("2.2 Linear space (vector space)")
    st.markdown(
        """
        A linear space is a set equipped with two fundamental operations:
        1. addition,
        2. multiplication by a scalar.

        For function spaces, this means:
        """
    )
    st.latex(r"f(x)+g(x)\in V,\qquad \alpha f(x)\in V\quad \text{for every scalar }\alpha")

    st.subheader("2.3 Why is it called linear?")
    st.markdown(
        """
        The term linear reflects the fact that we only combine elements by adding them
        and scaling them. We do not generally require closure under squaring, multiplying
        elements together, or composing functions.
        """
    )

    st.subheader("2.4 Linear combination")
    st.latex(r"v=a_1v_1+\cdots+a_nv_n")
    st.latex(r"f(x)=a_1\phi_1(x)+\cdots+a_n\phi_n(x)")

    st.subheader("2.5 Span")
    st.markdown("The span of a family of functions is the set of all linear combinations generated from them.")
    st.latex(r"\operatorname{span}\{\phi_1,\ldots,\phi_n\}=\{a_1\phi_1+\cdots+a_n\phi_n:\ a_i\in\mathbb{R}\}")

    st.subheader("2.6 Linear independence")
    st.latex(r"a_1\phi_1(x)+\cdots+a_n\phi_n(x)=0\ \forall x \quad\Longrightarrow\quad a_1=\cdots=a_n=0")

    st.subheader("2.7 Basis")
    st.markdown(
        """
        A basis is a smallest non-redundant generating family:
        it spans the space and is linearly independent.
        """
    )

    st.subheader("2.8 Dimension")
    st.markdown("The dimension of a finite-dimensional linear space is the number of elements in a basis.")

    st.subheader("2.9 Finite-dimensional vs infinite-dimensional")
    st.markdown(
        """
        A function space is finite-dimensional if a finite basis exists.
        This is crucial for MBM because it reduces infinitely many possible functions
        to finitely many coordinates.
        """
    )

    st.subheader("2.10 Closure under differentiation")
    st.markdown(
        """
        The key structural property for MBM is closure under differentiation:
        """
    )
    st.latex(r"f(x)\in V \quad\Longrightarrow\quad f'(x)\in V")
    st.markdown(
        """
        When this holds, differentiation acts internally inside the same finite-dimensional
        space. This is what makes the matrix representation possible.
        """
    )

# ============================================================
# 3) Finite-Dimensional Closure and MBM Representation
# ============================================================
elif section == "Finite-Dimensional Closure and the MBM Representation":
    st.header("3. Finite-Dimensional Closure and the MBM Representation")

    st.subheader("3.1 Vector representation of functions")
    st.markdown("Let the basis of the finite-dimensional space be")
    st.latex(r"\{\phi_1(x),\phi_2(x),\ldots,\phi_n(x)\}")

    st.markdown("Define the basis vector")
    st.latex(r"\Phi(x)=\begin{bmatrix}\phi_1(x)\\ \phi_2(x)\\ \vdots\\ \phi_n(x)\end{bmatrix}")

    st.markdown("Any function in the space can be written as")
    st.latex(r"f(x)=u^{\mathsf T}\Phi(x)")
    st.latex(r"u=\begin{bmatrix}a_1\\ a_2\\ \vdots\\ a_n\end{bmatrix}")

    st.subheader("3.2 Matrix representation of differentiation")
    st.markdown("If the space is closed under differentiation, then each basis derivative is a linear combination of the basis.")
    st.latex(r"\phi_i'(x)=\sum_{j=1}^n \Omega_{ji}\phi_j(x)")
    st.latex(r"\Phi'(x)=\Omega\Phi(x)")

    st.subheader("3.3 Structured integral problem")
    st.latex(r"\int e^{bx}f(x)\,dx")
    st.latex(r"f(x)=u^{\mathsf T}\Phi(x)")

    st.subheader("3.4 Motivation from integration by parts")
    st.markdown(
        """
        Classical repeated integration by parts repeatedly produces terms of the same
        structural kind. This suggests that the antiderivative remains inside the same
        structured family.
        """
    )

    st.subheader("3.5 Candidate antiderivative (MBM ansatz)")
    st.latex(r"F(x)=e^{bx}\Phi(x)^{\mathsf T}z")

    st.subheader("3.6 Differentiation of the candidate expression")
    st.latex(r"F'(x)=\frac{d}{dx}\left(e^{bx}\Phi(x)^{\mathsf T}z\right)")
    st.latex(r"=be^{bx}\Phi(x)^{\mathsf T}z + e^{bx}(\Phi'(x))^{\mathsf T}z")
    st.latex(r"=be^{bx}\Phi(x)^{\mathsf T}z + e^{bx}(\Omega\Phi(x))^{\mathsf T}z")
    st.latex(r"=be^{bx}\Phi(x)^{\mathsf T}z + e^{bx}\Phi(x)^{\mathsf T}\Omega^{\mathsf T}z")
    st.latex(r"=e^{bx}\Phi(x)^{\mathsf T}(bI+\Omega^{\mathsf T})z")

    st.subheader("3.7 Matching the derivative with the integrand")
    st.latex(r"e^{bx}f(x)=e^{bx}u^{\mathsf T}\Phi(x)=e^{bx}\Phi(x)^{\mathsf T}u")
    st.latex(r"e^{bx}\Phi(x)^{\mathsf T}(bI+\Omega^{\mathsf T})z=e^{bx}\Phi(x)^{\mathsf T}u")
    st.latex(r"(bI+\Omega^{\mathsf T})z=u")
    st.latex(r"z=(bI+\Omega^{\mathsf T})^{-1}u")

    st.subheader("3.8 Closed-form result")
    st.latex(r"\int e^{bx}f(x)\,dx=e^{bx}\Phi(x)^{\mathsf T}(bI+\Omega^{\mathsf T})^{-1}u + C")

    st.subheader("3.9 Finite interval version")
    st.latex(
        r"\int_0^a e^{bx}f(x)\,dx="
        r"\left[e^{bx}\Phi(x)^{\mathsf T}(bI+\Omega^{\mathsf T})^{-1}u\right]_0^a"
    )

# ============================================================
# 4) Operator interpretation and recipe
# ============================================================
elif section == "Operator Interpretation and Basic Laws":
    st.header("4. Operator Interpretation and Basic Laws")

    st.subheader("4.1 Scalar operator identity")
    st.markdown("Start from the first-order differential equation")
    st.latex(r"y'(x)+by(x)=f(x)")
    st.latex(r"(D+b)y=f,\qquad D=\frac{d}{dx}")

    st.markdown("The inverse operator is represented causally by")
    st.latex(r"(D+b)^{-1}f(x)=e^{-bx}\int_0^x e^{bs}f(s)\,ds")

    st.markdown("Multiplying by the exponential factor gives")
    st.latex(r"e^{bx}(D+b)^{-1}f(x)=\int_0^x e^{bs}f(s)\,ds")

    st.markdown("Evaluating at x=a yields")
    st.latex(r"\int_0^a e^{bx}f(x)\,dx=\left[e^{bx}(D+b)^{-1}f(x)\right]_0^a")

    st.subheader("4.2 Restriction to a finite-dimensional closed-under-differentiation space")
    st.latex(r"f(x)=u^{\mathsf T}\Phi(x),\qquad \Phi'(x)=\Omega\Phi(x)")
    st.latex(r"(D+b)^{-1}f(x)=\Phi(x)^{\mathsf T}(bI+\Omega^{\mathsf T})^{-1}u")

    st.markdown("Thus the matrix")
    st.latex(r"(bI+\Omega^{\mathsf T})^{-1}")
    st.markdown(
        "is the finite-dimensional representation of the inverse operator "
        r"\((D+b)^{-1}\) on the closed state space."
    )

    st.subheader("4.3 Linearity")
    st.latex(
        r"\int_0^a e^{bx}\left(\sum_j c_j f_j(x)\right)\,dx"
        r"=\sum_j c_j\left[e^{bx}\Phi(x)^{\mathsf T}M(b)u_j\right]_0^a"
    )
if section == "Operator Interpretation and Basic Laws":
    st.header("Operator interpretation and basic laws")
    st.subheader("Recipe (five steps)")

    st.markdown("**1. Choose a basis spanning the function.**")
    st.latex(r"f(x)=u^{\mathsf T}\Phi(x)")

    st.markdown("**2. Build the generator matrix.**")
    st.latex(r"\Phi'(x)=\Omega\Phi(x)")

    st.markdown("**3. Invert once.**")
    st.latex(r"M(b)=(bI+\Omega^{\mathsf T})^{-1}")

    st.markdown("**4. Form the boundary primitive.**")
    st.latex(r"F(x)=e^{bx}\Phi(x)^{\mathsf T}M(b)u")

    st.markdown("**5. Evaluate the definite integral.**")
    st.latex(r"\int_0^a e^{bx}f(x)\,dx = F(a)-F(0)")
# ============================================================
# import streamlit as st
import numpy as np
import math

# 1. First, Define all helper functions at the top
def rising_factorial(i, k):
    result = 1
    for j in range(k):
        result *= (i + j)
    return result

def Explicit_factorial_matrix(n):
    rows = []
    for row in range(n + 1):
        row_entries = []
        start_factor = row + 1
        for col in range(n + 1):
            if col < row:
                row_entries.append("0")
            else:
                k = col - row
                if k == 0:
                    row_entries.append(r"\frac{1}{t}")
                else:
                    factors = [str(start_factor + m) for m in range(k)]
                    product_part = r"\cdot".join(factors)
                    numerator = "(" + product_part + ")" + r"\," + str(k) + "!"
                    term = r"\frac{" + numerator + r"}{t^{" + str(k + 1) + "}}"
                    if k % 2 == 1:
                        term = "-" + term
                    row_entries.append(term)
        rows.append(row_entries)
    body = r"\\ ".join([" & ".join(r) for r in rows])
    return r"\begin{bmatrix}" + body + r"\end{bmatrix}"

# 2. Main Logic: Check if we are in the correct section
if section == "Canonical Bases and Ready Matrices":
    st.header("Canonical Bases and Ready Matrices")

    # --- Sections 5.1 to 5.4 ---
    st.subheader("5.1 Trigonometric basis")
    st.latex(r"\Phi(x)=\begin{bmatrix}\cos(\omega x)\\ \sin(\omega x)\end{bmatrix}")
    
    st.subheader("5.2 Polynomial basis")
    st.latex(r"\Phi(x)=\begin{bmatrix}1\\x\\x^2\end{bmatrix}")

    st.subheader("5.3 Exp–Trig basis")
    st.latex(r"\Phi(x)=\begin{bmatrix}e^{\alpha x}\cos(\omega x)\\ e^{\alpha x}\sin(\omega x)\end{bmatrix}")

    st.subheader("5.4 Exp–Hyp basis")
    st.latex(r"\Phi(x)=\begin{bmatrix}e^{\alpha x}\cosh(\omega x)\\ e^{\alpha x}\sinh(\omega x)\end{bmatrix}")

    # --- Section 5.5 ---
    st.subheader("5.5 Diagonal-Laplace Reading for Polynomial Bases")
    st.markdown("In the ordered polynomial basis, the resolvent matrix is upper triangular and may be read diagonally.")

    st.markdown("### Diagonal rule")
    st.latex(r"M_{i,i+k}=(-1)^k(i)_k\frac{k!}{t^{k+1}}")

    # --- This is where we place the Matrix Expander ---
    with st.expander("Show Matrix Details for polynomials"):
        n_diag_input = st.number_input("Enter Matrix Size (n):", min_value=1, max_value=10, value=3, key="n_input_55")
        
        st.markdown("#### Explicit factorial matrix pattern")
        matrix_latex = Explicit_factorial_matrix(n_diag_input)
        st.latex(r"M_{\text{Explicit}}=" + matrix_latex)

# 3. Handle other sections to prevent them from showing the matrix
elif section == "Operator Interpretation and Basic Laws":
    st.header("Operator Interpretation and Basic Laws")
    st.write("Content for this section...")

elif section == "Worked Examples":
    st.header("Worked Examples")
    st.write("Examples content...")

#============================================================
# 6) Worked examples
# ===========================================================

if section == "Worked Examples for Integrals":
    st.header("6. Worked Examples for Integrals")

    st.subheader("6.1 Trig kernel")
    st.latex(
        r"\int_0^a e^{bx}(u\cos(\omega x)+v\sin(\omega x))\,dx"
        r"=\left[e^{bx}\Phi(x)^{\mathsf T}M_0(b,\omega)\begin{bmatrix}u\\v\end{bmatrix}\right]_0^a"
    )

    st.subheader("6.2 Polynomial example")
    st.latex(r"f(x)=x^2-2x,\qquad u=\begin{bmatrix}0\\-2\\1\end{bmatrix}")
    st.latex(r"\int_0^a e^{bx}(x^2-2x)\,dx=e^{bx}(1,x,x^2)M(b)u\Big|_0^a")

    st.subheader("6.3 Shift property")
    st.latex(r"\int_0^a e^{bx}e^{\alpha x}g(x)\,dx=\int_0^a e^{(b+\alpha)x}g(x)\,dx")
    st.latex(r"M(b)\longrightarrow M(b+\alpha)")

    st.subheader("6.4 Example: no integration by parts")
    st.latex(r"f(x)=x\cos(\omega x)")
    st.latex(r"\Phi(x)=\begin{bmatrix}\cos(\omega x)\\ \sin(\omega x)\\ x\cos(\omega x)\\ x\sin(\omega x)\end{bmatrix}")
    st.latex(r"\Omega=\begin{bmatrix}0&-\omega&0&0\\ \omega&0&0&0\\ 1&0&0&-\omega\\ 0&1&\omega&0\end{bmatrix}")

    st.subheader("6.5 Example: second-order trig-polynomial block")
    st.latex(r"f(x)=x^2\cos(\omega x)")
    st.latex(r"\Phi(x)=\begin{bmatrix}\cos(\omega x)\\ \sin(\omega x)\\ x\cos(\omega x)\\ x\sin(\omega x)\\ x^2\cos(\omega x)\\ x^2\sin(\omega x)\end{bmatrix}")

# ============================================================
# 7) Products
# ============================================================
elif section == "Generalization: MBM for Products":
    st.header("7. Generalization: MBM for Products")

    st.markdown("Suppose")
    st.latex(r"f(x)=u^{\mathsf T}\Phi_f(x),\qquad G(x)=v^{\mathsf T}\Phi_g(x)")
    st.latex(r"\Phi_f'(x)=\Omega_f\Phi_f(x),\qquad \Phi_g'(x)=\Omega_g\Phi_g(x)")

    st.markdown("Define the joint state by the Kronecker product")
    st.latex(r"\Psi(x)=\Phi_g(x)\otimes \Phi_f(x)")
    st.latex(r"\Psi'(x)=\big(\Omega_g\otimes I + I\otimes \Omega_f\big)\Psi(x)")

    st.latex(r"w=v\otimes u")
    st.latex(r"f(x)G(x)=w^{\mathsf T}\Psi(x)")

    st.latex(
        r"\int_0^a e^{bx}f(x)G(x)\,dx"
        r"=\left[e^{bx}\Psi(x)^{\mathsf T}M_{\otimes}(b)w\right]_0^a"
    )
    st.latex(r"M_{\otimes}(b)=\big(bI+\Omega_g\otimes I + I\otimes \Omega_f\big)^{-1}")

    st.markdown("In practice, solve the linear system directly instead of forming the full inverse.")

# ============================================================
# 8) Exponential-matrix variant
# ============================================================
elif section == "Exponential-Matrix Variant":
    st.header("8. Exponential-Matrix Variant (New Insight)")

    st.markdown("We consider the constant-matrix differential system")
    st.latex(r"\Phi'(x)=\Omega\Phi(x)")
    st.markdown("and claim that")
    st.latex(r"\Phi(x)=e^{x\Omega}\Phi(0)")

    with st.expander("Detailed proof via the matrix-exponential series", expanded=True):
        st.markdown("**Step 1: Define the matrix exponential**")
        st.latex(
            r"e^{x\Omega}=I+x\Omega+\frac{x^2\Omega^2}{2!}+\frac{x^3\Omega^3}{3!}+\cdots"
            r"=\sum_{n=0}^{\infty}\frac{x^n\Omega^n}{n!}"
        )

        st.markdown("**Step 2: Differentiate term by term**")
        st.latex(
            r"\frac{d}{dx}e^{x\Omega}"
            r"=\frac{d}{dx}\left(I+x\Omega+\frac{x^2\Omega^2}{2!}+\frac{x^3\Omega^3}{3!}+\cdots\right)"
        )
        st.latex(
            r"=0+\Omega+\frac{2x\Omega^2}{2!}+\frac{3x^2\Omega^3}{3!}+\cdots"
        )
        st.latex(
            r"=\Omega+x\Omega^2+\frac{x^2\Omega^3}{2!}+\frac{x^3\Omega^4}{3!}+\cdots"
        )
        st.latex(
            r"=\Omega\left(I+x\Omega+\frac{x^2\Omega^2}{2!}+\frac{x^3\Omega^3}{3!}+\cdots\right)"
        )
        st.latex(r"\frac{d}{dx}e^{x\Omega}=\Omega e^{x\Omega}")

        st.markdown("**Step 3: Define the proposed solution**")
        st.latex(r"\Phi(x)=e^{x\Omega}\Phi(0)")

        st.markdown("**Step 4: Differentiate the proposed solution**")
        st.latex(r"\Phi'(x)=\frac{d}{dx}\big(e^{x\Omega}\Phi(0)\big)")
        st.latex(r"=\left(\frac{d}{dx}e^{x\Omega}\right)\Phi(0)")
        st.latex(r"=\Omega e^{x\Omega}\Phi(0)")
        st.latex(r"=\Omega\Phi(x)")

        st.markdown("**Step 5: Verify the initial condition**")
        st.latex(r"\Phi(0)=e^{0\Omega}\Phi(0)=I\Phi(0)=\Phi(0)")

        st.markdown("**Conclusion**")
        st.latex(r"\Phi(x)=e^{x\Omega}\Phi(0)")

    st.subheader("Appendix B: two proofs of the exponential-matrix integral formula")
    st.latex(
        r"I(b):=\int_0^a e^{-bx}f(x)\,dx"
        r"=u^{\mathsf T}(bI-\Omega)^{-1}\left(I-e^{-a(bI-\Omega)}\right)\Phi(0)"
    )

# ============================================================
# 9) Laplace connection
# ============================================================
elif section == "Laplace Connection":
    st.header("9. Laplace Connection")

    st.subheader("9.1 Matrix-form truncated Laplace transform")
    st.latex(r"I(s;a)=\int_0^a e^{-sx}f(x)\,dx")
    st.latex(r"f(x)=u^{\mathsf T}\Phi(x),\qquad \Phi'(x)=\Omega\Phi(x)")
    st.latex(
        r"I(s;a)=u^{\mathsf T}(sI-\Omega)^{-1}\left(I-e^{-a(sI-\Omega)}\right)\Phi(0)"
    )

    st.subheader("9.2 Laplace limit")
    st.latex(r"\lim_{a\to\infty}I(s;a)=u^{\mathsf T}(sI-\Omega)^{-1}\Phi(0)")
    st.latex(r"\mathcal{L}\{f\}(s)=u^{\mathsf T}(sI-\Omega)^{-1}\Phi(0)")

    st.subheader("9.3 Classical examples")
    st.latex(r"\mathcal{L}\{\cos(\omega x)\}(s)=\frac{s}{s^2+\omega^2}")
    st.latex(r"\mathcal{L}\{\sin(\omega x)\}(s)=\frac{\omega}{s^2+\omega^2}")
    st.latex(r"\mathcal{L}\{e^{\alpha x}\}(s)=\frac{1}{s-\alpha}")

# ============================================================
# 10) ODE via MBM
# ============================================================
elif section == "Part II: Linear ODEs via MBM":
    st.header("10. Part II: Linear ODEs via MBM")

    st.subheader("10.1 From convolution to boundary")
    st.latex(r"y'(x)-by(x)=f(x),\qquad y(0)=y_0")
    st.latex(r"y(x)=e^{bx}y_0+e^{bx}\int_0^x e^{-bs}f(s)\,ds")

    st.markdown("If")
    st.latex(r"f(s)=u^{\mathsf T}\Phi(s),\qquad \Phi'(s)=\Omega\Phi(s)")
    st.markdown("then")
    st.latex(
        r"y(x)=e^{bx}y_0+e^{bx}\left[e^{-bs}\Phi(s)^{\mathsf T}M(-b)u\right]_0^x"
    )

    st.subheader("10.2 Ready kernels")
    st.latex(r"\text{Trig: }\Phi=\begin{bmatrix}\cos(ts)\\\sin(ts)\end{bmatrix}")
    st.latex(r"M(\beta,t)=\frac1{\beta^2+t^2}\begin{bmatrix}\beta&-t\\ t&\beta\end{bmatrix}")

    st.latex(r"\text{Exp-Trig: }\Phi=\begin{bmatrix}e^{\alpha s}\cos(ts)\\e^{\alpha s}\sin(ts)\end{bmatrix}")
    st.latex(r"M(\beta)=\frac1{(\beta+\alpha)^2+t^2}\begin{bmatrix}\beta+\alpha&-t\\ t&\beta+\alpha\end{bmatrix}")

    st.latex(r"\text{Poly: }\Phi=\begin{bmatrix}1\\s\\s^2\end{bmatrix}")

# ============================================================
# Family chooser for calculators
# 
#==================================================
def choose_family(prefix=""):
    family = st.selectbox(
        "Choose function family",
        [
            "Polynomial P(x)",
            "Trig u cos(wx) + v sin(wx)",
            "Exp-Trig e^(alpha x)(u cos(wx) + v sin(wx))",
            "x cos(wx)",
            "x^2 cos(wx)",
        ],
        key=f"family_{prefix}"
    )

    x = sp.Symbol("x")

    if family == "Polynomial P(x)":
        poly_str = st.text_input("Enter polynomial P(x)", "x", key=f"{prefix}_poly")
        poly_expr = sp.expand(sp.sympify(poly_str))

        st.latex(r"P(x)=" + sp.latex(poly_expr))

        poly = sp.Poly(poly_expr, x)
        deg = poly.degree()

        basis = [x**k for k in range(deg + 1)]
        basis_text = r"\Phi(x)=" + sp.latex(sp.Matrix(basis))

        Phi = [sp.lambdify(x, b, "numpy") for b in basis]

        Omega = np.zeros((deg + 1, deg + 1), dtype=float)
        for k in range(1, deg + 1):
            Omega[k , k- 1] = k

        coeffs = [float(poly.coeff_monomial(x**k)) for k in range(deg + 1)]
        u = np.array(coeffs, dtype=float).reshape(-1, 1)

        phi0 = np.zeros((deg + 1, 1))
        phi0[0, 0] = 1.0

        return family, Phi, Omega, u, basis_text, phi0

    elif family == "Trig u cos(wx) + v sin(wx)":
        u0 = st.number_input("u (cos coefficient)", value=1.0, key=f"u0_{prefix}")
        v0 = st.number_input("v (sin coefficient)", value=0.0, key=f"v0_{prefix}")
        w = st.number_input("w (frequency)", value=2.0, key=f"w_{prefix}")

        Phi, Omega, u = build_trig_basis(u0, v0, w)
        basis_text = r"\Phi(x)=\begin{bmatrix}\cos(\omega x)\\ \sin(\omega x)\end{bmatrix}"
        phi0 = Phi(0.0)

        return family, Phi, Omega, u, basis_text, phi0

    elif family == "Exp-Trig e^(alpha x)(u cos(wx) + v sin(wx))":
        alpha = st.number_input("alpha", value=1.0, key=f"alpha_{prefix}")
        u0 = st.number_input("u (cos coefficient)", value=1.0, key=f"u1_{prefix}")
        v0 = st.number_input("v (sin coefficient)", value=0.0, key=f"v1_{prefix}")
        w = st.number_input("w (frequency)", value=2.0, key=f"w1_{prefix}")

        Phi, Omega, u = build_exp_trig_basis(alpha, u0, v0, w)
        basis_text = r"\Phi(x)=\begin{bmatrix}e^{\alpha x}\cos(\omega x)\\ e^{\alpha x}\sin(\omega x)\end{bmatrix}"
        phi0 = Phi(0.0)

        return family, Phi, Omega, u, basis_text, phi0

    elif family == "x cos(wx)":
        w = st.number_input("w (frequency)", value=2.0, key=f"wxcos_{prefix}")

        Phi, Omega, u = build_xcos_basis(w)
        basis_text = r"\Phi(x)=\begin{bmatrix}\cos(\omega x)\\ \sin(\omega x)\\ x\cos(\omega x)\\ x\sin(\omega x)\end{bmatrix}"
        phi0 = Phi(0.0)

        return family, Phi, Omega, u, basis_text, phi0

    elif family == "x^2 cos(wx)":
        w = st.number_input("w (frequency)", value=2.0, key=f"wx2cos_{prefix}")

        Phi, Omega, u = build_x2cos_basis(w)
        basis_text = r"\Phi(x)=\begin{bmatrix}\cos(\omega x)\\ \sin(\omega x)\\ x\cos(\omega x)\\ x\sin(\omega x)\\ x^2\cos(\omega x)\\ x^2\sin(\omega x)\end{bmatrix}"
        phi0 = Phi(0.0)

        return family, Phi, Omega, u, basis_text, phi0
# ============================================================
# 11) Integral calculator
# ===================================================
def mbm_integral(Phi, Omega, u, a, b):
    n = Omega.shape[0]
    A = b * np.eye(n) + Omega.T
    z = np.linalg.solve(A, u)

    def F(x):
        if callable(Phi):
            phi_x = np.array(Phi(x), dtype=float).reshape(-1, 1)
        else:
            phi_x = np.array([[f(x)] for f in Phi], dtype=float)

        return float(np.exp(b * x) * (phi_x.T @ z)[0, 0])

    value = F(a) - F(0.0)
    return value, A, z, F

#==================================================
# 11) Integral calculator
#
# ============================================================
if section == "Integral Calculator":
    st.header("11. Integral Calculator")
    st.markdown("This calculator computes the structured integral")
    st.latex(r"\int_0^a e^{bx}f(x)\,dx")
    st.markdown("with the full MBM workflow.")

    family, Phi, Omega, u, basis_text, phi0 = choose_family(prefix="int")

    a = st.number_input("Upper limit a", value=2.0, key="a_int")
    b = st.number_input("Weight b", value=0.0, key="b_int")

    if st.button("Compute Integral"):
        value, A, z, F = mbm_integral(Phi, Omega, u, a, b)

        st.subheader("Detailed MBM solution")

        st.latex(r"\textbf{Step 1: Choose the basis } \Phi(x)")
        st.latex(basis_text)

        st.latex(r"\textbf{Step 2: Build the generator matrix } \Omega")
        st.latex(r"\Omega=" + latex_matrix(Omega))

        st.latex(r"\textbf{Step 3: Represent the function as } f(x)=u^{\mathsf T}\Phi(x)")
        st.latex(r"u=" + latex_vector(u))

        st.latex(r"\textbf{Step 4: Solve the linear system } (bI+\Omega^{\mathsf T})z=u")
        st.latex(r"bI+\Omega^{\mathsf T}=" + latex_matrix(A))
        st.latex(r"z=" + latex_vector(z))

        st.latex(r"\textbf{Step 5: Form the boundary primitive}")
        st.latex(r"F(x)=e^{bx}\Phi(x)^{\mathsf T}z")

        st.latex(r"\textbf{Step 6: Evaluate the interval boundary}")
        st.latex(rf"I=F({a})-F(0)={value:.16g}")

        st.success(f"Final integral value = {value:.16g}")
#===================================
def choose_laplace_family():
    lap_family = st.selectbox(
        "Choose Laplace function",
        [
            "1",
            "x",
            "x^2",
            "cos(wx)",
            "sin(wx)",
            "u cos(wx) + v sin(wx)",
            "e^(alpha x)",
            "e^(alpha x) cos(wx)",
            "e^(alpha x) sin(wx)",
        ],
        key="laplace_family_only"
    )

    x = sp.Symbol("x")

    if lap_family == "1":
        basis_text = r"\Phi(x)=\begin{bmatrix}1\end{bmatrix}"
        Phi = [lambda t: 1.0]
        Omega = np.array([[0.0]], dtype=float)
        u = np.array([[1.0]], dtype=float)
        phi0 = np.array([[1.0]], dtype=float)
        symbolic_laplace = r"\frac{1}{s}"
        return lap_family, Phi, Omega, u, basis_text, phi0, symbolic_laplace

    elif lap_family == "x":
        basis_text = r"\Phi(x)=\begin{bmatrix}1\\x\end{bmatrix}"
        Phi = [lambda t: 1.0, lambda t: t]
        Omega = np.array([[0.0, 0.0],
                          [1.0, 0.0]], dtype=float)
        u = np.array([[0.0], [1.0]], dtype=float)
        phi0 = np.array([[1.0], [0.0]], dtype=float)
        symbolic_laplace = r"\frac{1}{s^2}"
        return lap_family, Phi, Omega, u, basis_text, phi0, symbolic_laplace

    elif lap_family == "x^2":
        basis_text = r"\Phi(x)=\begin{bmatrix}1\\x\\x^2\end{bmatrix}"
        Phi = [lambda t: 1.0, lambda t: t, lambda t: t**2]
        Omega = np.array([[0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0],
                          [0.0, 2.0, 0.0]], dtype=float)
        u = np.array([[0.0], [0.0], [1.0]], dtype=float)
        phi0 = np.array([[1.0], [0.0], [0.0]], dtype=float)
        symbolic_laplace = r"\frac{2}{s^3}"
        return lap_family, Phi, Omega, u, basis_text, phi0, symbolic_laplace

    elif lap_family == "cos(wx)":
        w = st.number_input("w (frequency)", value=1.0, key="lap_w_cos")
        basis_text = r"\Phi(x)=\begin{bmatrix}\cos(\omega x)\\ \sin(\omega x)\end{bmatrix}"
        Phi = [
            lambda t, w=w: np.cos(w * t),
            lambda t, w=w: np.sin(w * t),
        ]
        Omega = np.array([[0.0, -w],
                          [w,  0.0]], dtype=float)
        u = np.array([[1.0], [0.0]], dtype=float)
        phi0 = np.array([[1.0], [0.0]], dtype=float)
        symbolic_laplace = rf"\frac{{s}}{{s^2+({w})^2}}"
        return lap_family, Phi, Omega, u, basis_text, phi0, symbolic_laplace

    elif lap_family == "sin(wx)":
        w = st.number_input("w (frequency)", value=1.0, key="lap_w_sin")
        basis_text = r"\Phi(x)=\begin{bmatrix}\cos(\omega x)\\ \sin(\omega x)\end{bmatrix}"
        Phi = [
            lambda t, w=w: np.cos(w * t),
            lambda t, w=w: np.sin(w * t),
        ]
        Omega = np.array([[0.0, -w],
                          [w,  0.0]], dtype=float)
        u = np.array([[0.0], [1.0]], dtype=float)
        phi0 = np.array([[1.0], [0.0]], dtype=float)
        symbolic_laplace = rf"\frac{{{w}}}{{s^2+({w})^2}}"
        return lap_family, Phi, Omega, u, basis_text, phi0, symbolic_laplace

    elif lap_family == "u cos(wx) + v sin(wx)":
        u0 = st.number_input("u (cos coefficient)", value=1.0, key="lap_u0")
        v0 = st.number_input("v (sin coefficient)", value=0.0, key="lap_v0")
        w = st.number_input("w (frequency)", value=1.0, key="lap_w_mix")
        basis_text = r"\Phi(x)=\begin{bmatrix}\cos(\omega x)\\ \sin(\omega x)\end{bmatrix}"
        Phi = [
            lambda t, w=w: np.cos(w * t),
            lambda t, w=w: np.sin(w * t),
        ]
        Omega = np.array([[0.0, -w],
                          [w,  0.0]], dtype=float)
        u = np.array([[u0], [v0]], dtype=float)
        phi0 = np.array([[1.0], [0.0]], dtype=float)
        symbolic_laplace = rf"\frac{{{u0}s+({v0*w})}}{{s^2+({w})^2}}"
        return lap_family, Phi, Omega, u, basis_text, phi0, symbolic_laplace

    elif lap_family == "e^(alpha x)":
        alpha = st.number_input("alpha", value=1.0, key="lap_alpha_exp")
        basis_text = r"\Phi(x)=\begin{bmatrix}e^{\alpha x}\end{bmatrix}"
        Phi = [lambda t, alpha=alpha: np.exp(alpha * t)]
        Omega = np.array([[alpha]], dtype=float)
        u = np.array([[1.0]], dtype=float)
        phi0 = np.array([[1.0]], dtype=float)
        symbolic_laplace = rf"\frac{{1}}{{s-({alpha})}}"
        return lap_family, Phi, Omega, u, basis_text, phi0, symbolic_laplace

    elif lap_family == "e^(alpha x) cos(wx)":
        alpha = st.number_input("alpha", value=1.0, key="lap_alpha_ecos")
        w = st.number_input("w (frequency)", value=1.0, key="lap_w_ecos")
        basis_text = r"\Phi(x)=\begin{bmatrix}e^{\alpha x}\cos(\omega x)\\ e^{\alpha x}\sin(\omega x)\end{bmatrix}"
        Phi = [
            lambda t, alpha=alpha, w=w: np.exp(alpha * t) * np.cos(w * t),
            lambda t, alpha=alpha, w=w: np.exp(alpha * t) * np.sin(w * t),
        ]
        Omega = np.array([[alpha, -w],
                          [w, alpha]], dtype=float)
        u = np.array([[1.0], [0.0]], dtype=float)
        phi0 = np.array([[1.0], [0.0]], dtype=float)
        symbolic_laplace = rf"\frac{{s-({alpha})}}{{(s-({alpha}))^2+({w})^2}}"
        return lap_family, Phi, Omega, u, basis_text, phi0, symbolic_laplace

    elif lap_family == "e^(alpha x) sin(wx)":
        alpha = st.number_input("alpha", value=1.0, key="lap_alpha_esin")
        w = st.number_input("w (frequency)", value=1.0, key="lap_w_esin")
        basis_text = r"\Phi(x)=\begin{bmatrix}e^{\alpha x}\cos(\omega x)\\ e^{\alpha x}\sin(\omega x)\end{bmatrix}"
        Phi = [
            lambda t, alpha=alpha, w=w: np.exp(alpha * t) * np.cos(w * t),
            lambda t, alpha=alpha, w=w: np.exp(alpha * t) * np.sin(w * t),
        ]
        Omega = np.array([[alpha, -w],
                          [w, alpha]], dtype=float)
        u = np.array([[0.0], [1.0]], dtype=float)
        phi0 = np.array([[1.0], [0.0]], dtype=float)
        symbolic_laplace = rf"\frac{{{w}}}{{(s-({alpha}))^2+({w})^2}}"
        return lap_family, Phi, Omega, u, basis_text, phi0, symbolic_laplace


#=================
# 12) Laplace calculator
# ============================================================
if section == "Laplace Calculator":
    st.header("12. Laplace Calculator")
    st.markdown("This section computes the classical Laplace transform only.")
    st.latex(r"\mathcal{L}\{f(x)\}(s)=\int_0^{\infty} e^{-sx}f(x)\,dx")

    family, Phi, Omega, u, basis_text, phi0, symbolic_laplace = choose_laplace_family()

    s = st.number_input("Laplace parameter s", value=2.0, min_value=0.1, key="s_lap")

    if st.button("Compute Laplace"):
        value, A = matrix_laplace_limit(Omega, u, phi0, s)

        st.subheader("Detailed Laplace solution")

        st.latex(r"\textbf{Step 1: Choose the basis}")
        st.latex(basis_text)

        st.latex(r"\textbf{Step 2: Build } \Omega")
        st.latex(r"\Omega=" + latex_matrix(Omega))

        st.latex(r"\textbf{Step 3: Compute the initial state } \Phi(0)")
        st.latex(r"\Phi(0)=" + latex_vector(phi0))

        st.latex(r"\textbf{Step 4: Build } sI-\Omega")
        st.latex(r"sI-\Omega=" + latex_matrix(A))

        st.latex(r"\textbf{Step 5: Apply the resolvent formula}")
        st.latex(r"\mathcal{L}\{f\}(s)=u^{\mathsf T}(sI-\Omega)^{-1}\Phi(0)")

        st.latex(r"\textbf{Closed-form result:}")
        st.latex(r"\mathcal{L}\{f\}(s)=" + symbolic_laplace)

        st.latex(rf"\textbf{{Numerical value at }} s={s}:")
        st.latex(rf"\mathcal{{L}}\{{f\}}({s})={value:.16g}")

        st.success(f"Laplace value at s = {s} is {value:.16g}")

# ===================================================
