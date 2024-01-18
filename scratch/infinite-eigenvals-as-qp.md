---
  title: Solving Systems with Infinite Eigenvalues
  author: Parth Nobel
...
\newcommand{\reals}{\mathbb{R}}

Consider the problem solve
$$
    (A + B) x = y
$$
where $A$, $B$ are symmetric and $B$ is allowed to have infinite eigenvalues.

An example is that the "Hessian" of $x \mapsto \|D x\|_1$ is $D^T \operatorname{diag}(f(x)) D$ where $f(x)_i = 0$ if $x_i \neq 0$ and $f(x)_i = \infty$ if $x_i = 0$.

We first decompose $B = B_\infty + B_\reals$ where $B_\reals$ has finite eigenvalues and $B_\infty$ has eigenvalues of $0, \infty$.
Let $U$ be a matrix whose row space is the orthogonal complement of the null space of $B_\infty$ (its "column space").

As an example, for the "Hessian" of  $x \mapsto \|D x\|_1$ $B_\reals = 0$ and $U = D^T\operatorname{diag}(g(x))D$ where $g(x)_i = 1$ if $x_i = 0$ and $g(x)_i = 0$ else.
Alternatively, $U$ is the right singular vectors of the columns of $D$ that correspond to zero entries in $x$.


For simplicity, assume $B_\infty, B_\reals, A$ are all symmetric.

Accordingly we note that $(A + B) x = y$ if and only if $(A + B_\reals) x = y'$, $B_\infty x = 0$, and $y' = \Pi y$, where $\Pi = I - B_\infty (B_\infty^T B_\infty)^\dagger B_\infty^T$.
The idea is that any component of $y$ in the column space of $B_\infty$, can be created with an infinitesimal mass in $x$, so we can remove those dimensions from $y$.
This is necessary, because $x$ cannot have non-infinitesimal mass outside the null space of $B_\infty$ and satisfy the first equation.
From here, we simplify this with $y' + B_\infty (B_\infty^T B_\infty)^\dagger B_\infty^T y'' = y $ and backsubstituting to get
$(A + B_\reals) x + B_\infty (B_\infty^T B_\infty)^\dagger B_\infty^T y'' = y$
Here we can note that since $\mathrm{Col}(B_\infty (B_\infty^T B_\infty)^\dagger B_\infty^T) = \mathrm{Col}(B_\infty^T)$, we have that $B_\infty (B_\infty^T B_\infty)^\dagger B_\infty^Ty'' = B_\infty^T \nu$ for some $\nu$ for any $y''$.

Therefore, we can write this as $(A + B_\reals) x + B_\infty^T \nu = y$ and $B_\infty x = 0$.
When $A + B_\reals$ are PSD, these are the optimality conditions of the following QP:
$$
\begin{array}{ll}
\text{minimize} & \frac{1}{2} x^T (A + B_\reals) x - x^T y \\
\text{subject to} & B_\infty x = 0
\end{array}.
$$

Substituting $B_\infty$ with a possibly reduced matrix $U$, we have
$$
\begin{array}{ll}
\text{minimize} & \frac{1}{2} x^T (A + B_\reals) x - x^T y \\
\text{subject to} & U x = 0
\end{array}.
$$

This can be solved by solving the following QSD system: 
$$
\begin{bmatrix}
A + B_\reals & U^T \\
U & 0
\end{bmatrix}
\begin{bmatrix}
x \\ \nu
\end{bmatrix}
=
\begin{bmatrix}
y \\ 0
\end{bmatrix},
$$
which can be solved with MINRES in many cases.
