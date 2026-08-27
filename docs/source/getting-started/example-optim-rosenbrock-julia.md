# Example: Minimize the Rosenbrock function using Julia

This tutorial shows how to use _Open Interfaces_ to
minimize the
[Rosenbrock function][rosenbrock-wiki] using Julia `optim` gateway.

The problem is mathematically defined as
finding the minimizing vector $x \in \mathbb{R}^n$
for the objective function

$$
  f(x) = \sum_{i=1}^{n-1} \left(a (x_{i+1} - x_i^2)^2
  + (1 - x_i)^2\right),
$$

where $a = 10$. The global minimizer is
$x = [1, 1, \ldots, 1]$, where $f(x) = 0$.

## Loading the interface

The example begins by importing the optimization interface:

```julia
using OpenInterfaces.Interfaces.Optim
```

The first call obtains an optimizer instance for a specific backend implementation:

```julia
s = Optim.Self(impl)
```

The `impl` string selects the optimization backend.
The supported values for the `optim` interface are:

- `optim_jl`
- `scipy_optimize`

## Defining the objective and gradient

Let's define the objective and gradient functions:

```julia
function rosenbrock_objective_fn(x, a)
    return sum(a * (x[2:end] - x[1:(end-1)] .^ 2.0) .^ 2.0 + (1 .- x[1:(end-1)]) .^ 2.0)
end

function rosenbrock_grad_fn(x, grad_f, a)
    xi = @view x[1:(end-1)]
    xip1 = @view x[2:end]

    grad_f[1:(end-1)] .= -4.0 .* a .* xi .* (xip1 .- xi .^ 2.0) .- 2.0 .* (1.0 .- xi)
    grad_f[end] = 0.0
    grad_f[2:end] .+= 2.0 .* a .* (xip1 .- xi .^ 2.0)
    return 0
end
```


## Configuring the optimizer

First of all we instantiate an instance of the `optim` interface.
As the Julia components use the object-oriented approach,
we instantiate the provided structure `Self`:

```julia
s = Optim.Self("scipy_optimize")
```
where we use `scipy_optimize` (Python implementation).

Next we need to set an initial guess for the nonlinear optimization process:

```julia
x0 = [3.14, 2.72, 6.18, 9.81, 8.31]

Optim.set_initial_guess(s, x0)
```

Our objective function takes an additional parameter $a$,
so we need to pass it as _user data_:
```julia
user_data = 10
Optim.set_user_data(s, user_data)
```

Now we can select the optimization method,
that we want to use and their respective parameters.
These parameters are always method-specific, and should be
looked for in the documentation for the given method.

For example, we can select the `NelderMead` algorithm,
which in SciPy allows to set the termination criterion
to be the norm of the objective function:

```julia
Optim.set_method(s, "nelder-mead", Dict("fatol" => 1e-11))
```

or we can select the `BFGS` method,
with the termination criterion defined as the inf-norm
of the gradient function:
```julia
Optim.set_method(s, "BFGS", Dict("gtol" => 1e-8))
```

The objective and gradient callbacks are then passed
to the implementations (note that the `Nelder-Mead` does not require
`set_grad_fn`, so it can be omitted):

```julia
Optim.set_objective_fn(s, rosenbrock_objective_fn)
Optim.set_grad_fn(s, rosenbrock_grad_fn)
```

## Running the minimization

Once the problem is fully configured, the minimization can be invoked:

```julia
status, message = Optim.minimize(s)
x = s.x

println("Message: ", message)
@assert status == 0
println("x = ", x)
```

Note that we use the convention that the `status` is equal to zero,
when the used solver reports successful termination.
Additional return argument `message` is used to pass a free-form message,
that different solver use to define more details about the minimization
(for example, they can report only local convergence,
not the global one).

A successful run finds a vector whose entries are all near `1.0`, which is the global minimum of the Rosenbrock function.

## Running the full script

The complete example is available at `examples/lang_julia/call_optim_rosenbrock.jl`.
Assuming the project has been built and the environment is loaded via `source env.sh`, run it from the project root with:

```shell
julia examples/lang_julia/call_optim_rosenbrock.jl [implementation] [method] [linesearch]
```

Supported combinations of the command-line arguments for this script are:

- `optim_jl NelderMead`
- `optim_jl BFGS StrongWolfe`
- `scipy_optimize NelderMead`
- `scipy_optimize BFGS`

The line-search argument is only used for the Julia `BFGS` implementation. The script prints the chosen backend, the method, the solver message, and the optimizer output, and then verifies that the result is close to the true minimizer.

[rosenbrock-wiki]: https://en.wikipedia.org/wiki/Rosenbrock_function
