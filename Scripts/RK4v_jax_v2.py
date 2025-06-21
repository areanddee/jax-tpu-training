import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial

# Define the function f(t, y) = y for our differential equation
def f(t, y):
    return y

# Define RK4 function but separate the JIT decoration
def rungekutta4_impl(t0, y0, h, nit):
    """
    Runge-Kutta 4th order method implemented with JAX
    
    Parameters:
    t0 : float
        Initial time
    y0 : jnp.ndarray
        Initial condition vector
    h : float
        Step size
    nit : int
        Number of iterations
    
    Returns:
    t_values : jnp.ndarray
        Time points
    y_values : jnp.ndarray
        Solution values at each time point
    """
    # Pre-allocate arrays for results
    t_values = jnp.zeros(nit + 1)
    y_values = jnp.zeros((nit + 1, y0.shape[0]))
    
    # Set initial conditions
    t_values = t_values.at[0].set(t0)
    y_values = y_values.at[0].set(y0)
    
    # Define a single step of RK4
    def rk4_step(carry, _):
        t, y = carry
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y_next = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        t_next = t + h
        return (t_next, y_next), (t_next, y_next)
    
    # Use scan to iterate through time steps
    (_, _), (ts, ys) = jax.lax.scan(
        rk4_step, 
        (t0, y0), 
        jnp.arange(nit)
    )
    
    # Fill in the results
    t_values = t_values.at[1:].set(ts)
    y_values = y_values.at[1:].set(ys)
    
    return t_values, y_values

# Create a jitted version with nit as a static argument
@partial(jax.jit, static_argnums=(3,))
def rungekutta4(t0, y0, h, nit):
    return rungekutta4_impl(t0, y0, h, nit)

# Set up the problem
t0 = 0.0                       # Initial time
npoints = 50                   # Number of spatial points
y0 = jnp.ones(npoints)         # Initial condition vector (all ones)
h = 0.1                        # Step size
nit = 100                      # Number of time steps

# Solve using RK4
t_values, y_values = rungekutta4(t0, y0, h, nit)

# Compute analytical solution y = e^t
analytical = jnp.exp(t_values.reshape(-1, 1)) * jnp.ones_like(y_values)

# Plot results
plt.figure(figsize=(10, 6))

# Plot numerical solution for a few points
plot_indices = [0, npoints//4, npoints//2, 3*npoints//4, npoints-1]
for idx in plot_indices:
    plt.plot(t_values, y_values[:, idx], '-', label=f'RK4 Point {idx}')

# Plot analytical solution
plt.plot(t_values, analytical[:, 0], 'k--', label='Analytical')

plt.xlabel('Time')
plt.ylabel('y')
plt.title('Vectorized RK4 Solution for dy/dt = y, y(0) = 1')
plt.legend()
plt.grid(True)
plt.show()
