# Example: Minimize the Rosenbrock function using Python

In this example, we use _Open Interfaces_ from Python to minimize the
[Rosenbrock function][rosenbrock-wiki]. For a vector $x \in \mathbb{R}^n$,
the objective function used by the example is

$$
  f(x) = \sum_{i=1}^{n-1} \left(a (x_{i+1} - x_i^2)^2
  + (1 - x_i)^2\right),
$$

where $a = 10$. The global minimizer is
$x = [1, 1, \ldots, 1]$, where $f(x) = 0$.

## Objective and gradient functions

The optimization interface calls the objective function with the current
vector and the user data. The example uses the user data to pass the parameter
$a$ without hard-coding it in the callback:

```python
def rosenbrock_objective_fn(x, a):
    return np.sum(a * (x[1:] - x[:-1] ** 2.0) ** 2.0
                  + (1.0 - x[:-1]) ** 2.0)
```

The gradient callback receives the current vector, an output array, and the
same user data. It fills the output array in place and returns `0` to indicate
successful computation:

```python
def rosenbrock_grad_fn(x, grad_f, a):
    xi = x[:-1]
    xip1 = x[1:]

    grad_f[:-1] = -4.0 * a * xi * (xip1 - xi**2.0) - 2.0 * (1.0 - xi)
    grad_f[-1] = 0.0
    grad_f[1:] += 2.0 * a * (xip1 - xi**2.0)
    return 0
```

The callback signatures are defined by the interface so that the same
optimization problem can be used by implementations written in different
languages.

## Configuring the optimizer

After importing NumPy and the optimization interface, define an initial guess
and create an optimizer. The example starts from some random initial guess
("random", but consisting of approximations of some constants):

```python
import numpy as np
from openinterfaces.interfaces.optim import Optim

x0 = np.array([3.14, 2.72, 6.18, 9.81, 8.31])
# Instantiate a Gateway to the `optim` interface
s = Optim("optim_jl")
s.set_initial_guess(x0)
```

Because our objective function has the user-defined parameter `a`,
we need to pass it as well:
```python
user_data = 10.0

s.set_user_data(user_data)
```


## Passing callbacks to the objective and gradient functions

```python
s.set_objective_fn(rosenbrock_objective_fn)
s.set_grad_fn(rosenbrock_grad_fn)
```

The `optim_jl` implementation uses optimization algorithms from the Julia
ecosystem.
For Nelder-Mead, we can find in the documentation
at <https://julianlsolvers.github.io/Optim.jl/stable/algo/nelder_mead/>,
that the absolute function tolerance is controlled
by the parameter `g_abstol` (or `g_tol`),
hence we define the termination criterion of the optimization process
as follows:

```python
s.set_method("NelderMead", {"g_abstol": 1e-11})
```

To use the BFGS method,
select the method, setting the termination criterion
to be the inf-norm of the gradient,
also setting the line-search strategy to `StrongWolfe`:

```python
s.set_method("BFGS", {"g_abstol": 1e-8, "linesearch": "StrongWolfe"})
```

Nelder-Mead is a gradient-free method,
so one does not need to pass the gradient callback.

For the BFGS, we pass the gradient callback
to improve the stability by using analytic gradient
instead of finite differences:
```python
s.set_grad_fn(rosenbrock_grad_fn)
```

So the full set of required calls to the `Optim`
interface to define an optimization problem
for, say, the gradient-based BFGS method
from the `Optim.jl` package is:

```python
s = Optim("optim_jl")
s.set_initial_guess(x0)
s.set_user_data(user_data)
s.set_method("BFGS", {"g_abstol": 1e-8, "linesearch": "StrongWolfe"})
s.set_objective_fn(rosenbrock_objective_fn)
s.set_grad_fn(rosenbrock_grad_fn)
```

## Minimizing

Now as the optimization problem is fully defined,
we are ready to call `minimize`.
Calling `minimize()` returns a status code and a message.
By convention, status code is equal to zero,
when the implementation reports a successful termination
of the optimization process.
The message is a string, which can be useful for the user
to see, as different solvers use free-form messages
to describe the status of optimization process
(for example, they can report that only local convergence is achieved,
but not the global one).

The resultant minimizer is available as the property `x`
of the `Optim` gateway:
```python
status, message = s.minimize()
assert status == 0

print(f"Message: {message}")
print(f"x = {s.x}")
```

For a successful run, every component of `s.x` is close to `1`
for the Rosenbrock function used in this example.

## Running the complete example

The full example is available in
`examples/lang_python/call_optim_rosenbrock.py`.
Assuming that _Open Interfaces_ are successfully built,
and the environment variables are set via `source env.sh`,
we can call the example from the root directory:
```shell
python examples/lang_python/call_optim_rosenbrock.py [implementation] [method]
[linesearch]
```

Supported combinations include:

 - `optim_jl NelderMead`
 - `optim_jl BFGS StrongWolfe`
 - `scipy_optimize NelderMead`
 - `scipy_optimize BFGS`

The line-search argument is used only by `optim_jl` with BFGS. The script
prints the implementation, method, solver message, and computed minimizer,
then checks that the result is close to the exact solution.


[rosenbrock-wiki]: https://en.wikipedia.org/wiki/Rosenbrock_function
