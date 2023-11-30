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

Accordingly we note that $A + B x = y$ if and only if $(A + B_\reals) x = y$ and $B_\infty x = 0$.
When $A + B_\reals$ are PSD, we can interpret this as the optimality conditions of the QP:
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
