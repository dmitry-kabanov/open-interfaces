optim
=====

.. py:module:: optim

.. autoapi-nested-parse::

   This module defines the interface for solving minimization problems:

   .. math::
       minimize_x f(x)

   where :math:`f : \mathbb R^n \to \mathbb R`.







Module Contents
---------------

.. py:type:: ObjectiveFn
   :canonical: Callable[[np.ndarray, object], int]


   Signature of the objective function :math:`f(, y)`.

   !!!!!!!!! The function accepts four arguments:
   !!!!!!!!!     - `t`: current time,
   !!!!!!!!!     - `y`: state vector at time :math:`t`,
   !!!!!!!!!     - `ydot`: output array to which the result
   !!!!!!!!!     - `user_data`: additional context (user-defined data) that
   !!!!!!!!!       must be passed to the function (e.g., parameters of the system).

.. py:class:: Optim(impl: str)

   Interface for solving optimization (minimization) problems.

   This class serves as a gateway to the implementations of the
   solvers for optimization problems.

   :param impl: Name of the desired implementation.
   :type impl: str

   .. rubric:: Examples

   Let's solve the following convex optimization problem:

   .. math::
       minimize \sum_{i = 1}^N x_i^2

   where the solution is :math:`[0, ..., 0]` as the problem is convex.

   First, import the necessary modules:
   >>> import numpy as np
   >>> from oif.interfaces.optim import Optim

   Define the objective function:

   >>> def objective_fn(x):
   ...     return np.sum(x**2)

   Create an instance of the optim solver using the implementation "scipy_optimize",
   which is an adapter to the `scipy.optimize` Python package:

   >>> s = Optim("scipy_optimize")

   We set the initial value, the right-hand side function, and the tolerance:

   >>> s.set_initial_guess([2.718, 3.142])
   >>> s.set_objective_fn(objective_fn)

   Now we solve the minimization problem and print the return status and message:

   >>> status, message = s.minimize()

   We can print the resultant minimizer by retrieving it from the solver:

   >>> print(f"Minimizer is {s.x}")



   .. py:attribute:: x0
      :type:  numpy.ndarray

      Current value of the state vector.


   .. py:attribute:: x
      :type:  numpy.ndarray


   .. py:attribute:: user_data
      :type:  object


   .. py:attribute:: oif_user_data
      :type:  openinterfaces.core.OIFUserData


   .. py:method:: set_initial_guess(x0: numpy.ndarray)

      Set initial guess for the optimization problem



   .. py:method:: set_objective_fn(objective_fn: ObjectiveFn)


   .. py:method:: set_grad_fn(grad_fn)


   .. py:method:: set_user_data(user_data: object)

      Specify additional data that will be used for right-hand side function.



   .. py:method:: set_method(method_name: str, method_params: dict = {})

      Set integrator, if the name is recognizable.



   .. py:method:: minimize()

      Integrate to time `t` and write solution to `y`.
