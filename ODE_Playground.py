#%%
import numpy as np
from world import World
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from ode_solver import Euler,RungeKutta45

#%% Exponential function
######################### EXPONENTIAL FUNCTION
def f(t,u): # exponential function
    return -2*u

###### Time axis
T = 5 
dt = 0.3
n = int(round(T/dt))
time_points = np.linspace(0,T,n+1)

### Using our solver to find solutions and plotting
plt.figure()
plt.title("ODE solutions to dy/dt = -u")
for solver_method in [RungeKutta45, Euler]:
    solver = solver_method(f)
    solver.set_initial_conditions(1)
    y,t = solver.solve(time_points)
    plt.plot(t,y, label = solver.name())

### plot exact solution
t_fine = np.linspace(0,T,10001)
plt.plot(t_fine, np.exp(-2*t_fine),label = "Exact solutin")
plt.legend()
plt.show()

#%% Sin function
######################### SIN(X) FUNCTION: dY/dt = sin(t)
def sin(t,u): # exponential function
    return -np.sin(t)

###### Time axis
T = 30 
dt = 0.5
n = int(round(T/dt))
time_points = np.linspace(0,T,n+1)

### Using our solver to find solutions and plotting
plt.figure()
plt.title("ODE solutions to dy/dt = -sin(t)")
for solver_method in [RungeKutta45, Euler]:
    solver = solver_method(sin)
    solver.set_initial_conditions(1)
    y,t = solver.solve(time_points)
    plt.plot(t,y, label = solver.name())

### plot exact solution
t_fine = np.linspace(0,T,10001)
plt.plot(t_fine, np.cos(-t_fine),label = "Exact solutin")
plt.legend()
plt.show()

#%% oscillator
######################### Oscillator : d^2 Y/dt^2 = mg - k Y - B dY/dt 
from ode_examples import Oscillator

periods = 4
T = periods*2*np.pi 
n = int(1000)
osc_1 = Oscillator(1,10,0.31,5)
initial_conditions = [1,0]
time_points = np.linspace(0,T,n+1)
plt.figure()
solver = RungeKutta45(osc_1)
solver.set_initial_conditions(initial_conditions)
yosc_1,t = solver.solve(time_points)
plt.plot(time_points,yosc_1[:,0])
plt.show()
