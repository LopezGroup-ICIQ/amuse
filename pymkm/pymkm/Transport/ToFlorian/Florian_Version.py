from firedrake import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
#plt.rc('text', usetex=True)
plt.rc('font', size=8)

# Mesh definition
numel = 5000
x_left, x_right = -1.0, 1.0
mesh = IntervalMesh(numel, x_left, x_right)

# Function space declaration
degree = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", degree)

# Trial and Test functions
#u = Function(V)
#u_k = Function(V)

u0, u1, u2, u3 = TrialFunction(V),TrialFunction(V),TrialFunction(V),TrialFunction(V)
u0_k, u1_k, u2_k, u3_k = TrialFunction(V),TrialFunction(V),TrialFunction(V),TrialFunction(V)

u0_sol, u1_sol, u2_sol, u3_sol = Function(V),Function(V),Function(V),Function(V)
u0_k_sol, u1_k_sol, u2_k_sol, u3_k_sol = Function(V),Function(V),Function(V),Function(V)

v0, v1, v2, v3 = TestFunction(V),TestFunction(V),TestFunction(V),TestFunction(V)

# Diffusion parameter
D0 = Constant(0.01) #Constant(1.e0)
D1 = Constant(0.01)
D2 = Constant(0.01)
D3 = Constant(0.01)
K = Constant(1e1)

# Source term
#f = Constant(0.0)
f0 = -K * u0_sol * u1_sol
f1 = -K * u0_sol * u1_sol
f2 = K * u0_sol * u1_sol
f3 = Constant(0.0)

# Advection
r = Constant(0.0)

# Neumann condition for reaction at catalyst surface
def g(u):
    K2 = -2.5e-1
    #return K2*u # Could be function of concentration or call the MKM here !
    return K2

# Initial condition
x = SpatialCoordinate(mesh)
expr0 = Constant(1.0) * exp(- 1000.0 * x[0] * x[0]) #Constant(0.0) #exp(- 000.0 * x[0] * x[0])  # An expression to the initial condition
expr1 = 0.0 * x[0]
ic0 = Function(V).interpolate(expr0)
ic1 = Function(V).interpolate(expr1)


# Essential boundary conditions
boundary_value = 0.0
boundary_value_u = 1.0
boundary_value_v = 1.0
bcs0 = DirichletBC(V, boundary_value_u, [1])  # Boundary condition in 1 and 2 marked bounds (left and right)
bcs1 = DirichletBC(V, boundary_value_v, [1])  # Boundary condition in 1 and 2 marked bounds (left and right)
bcs2 = DirichletBC(V, boundary_value, [1,2])  # Boundary condition in 1 and 2 marked bounds (left and right)
bcs3 = DirichletBC(V, boundary_value, [1,2])  # Boundary condition in 1 and 2 marked bounds (left and right)


# Time step
dt = 0.01

# Assigning the IC
u0_k_sol.assign(ic1)
u0_sol.assign(ic1)

u1_k_sol.assign(ic1)
u1_sol.assign(ic1)

u2_k_sol.assign(ic1)
u2_sol.assign(ic1)

u3_k_sol.assign(ic1)
u3_sol.assign(ic1)

# Residual variational formulation
### Specie 0
F0 = inner((u0 - u0_k_sol) / dt, v0) * dx + inner(D0 * grad(u0), grad(v0)) * dx - inner(r * u0, v0) * dx
F0 -= f0 * v0 * dx
F0 += g(u0) * v0 * ds(2) + Constant(0.0) * v0 * ds(1)

a0, L0 = lhs(F0), rhs(F0)

### Specie 1
F1 = inner((u1 - u1_k_sol) / dt, v1) * dx + inner(D1 * grad(u1), grad(v1)) * dx - inner(r * u1, v1) * dx
F1 -= f1 * v1 * dx

a1, L1 = lhs(F1), rhs(F1)

### Specie 2
F2 = inner((u2 - u2_k_sol) / dt, v2) * dx + inner(D2 * grad(u2), grad(v2)) * dx - inner(r * u2, v2) * dx
F2 -= f2 * v2 * dx

a2, L2 = lhs(F2), rhs(F2)

### Specie 2
F3 = inner((u3 - u3_k_sol) / dt, v3) * dx + inner(D3 * grad(u3), grad(v3)) * dx - inner(r * u3, v3) * dx
F3 -= f3 * v3 * dx

a3, L3 = lhs(F3), rhs(F3)

# Convergence criteria
norm_l2 = 1.0e5  # Any arbitrary value greater than the tolerance
tolerance = 1.e-5

# Setting PETSc parameters and method to use a Direct Method (LU), valid for symmetric systems (be aware)
solver_parameters = {
    "ksp_type": "preonly",  # This set the method to perform only the preconditioner (LU, in the case)
    "pc_type": "lu"  # The desired preconditioner (LU)
}

# Iterating and solving over the time
t = dt
T_total = 1.0
step = 0
plot_step_mod = 1
x_values = mesh.coordinates.vector().dat.data
u0_sol_values = []
u1_sol_values = []
u2_sol_values = []
u3_sol_values = []
while t < T_total and norm_l2 > tolerance:
    step += 1
    print('============================')
    print('\ttime =', t)
    print('\tstep =', step)
    print('============================')
    solve(a0 == L0, u0_sol, bcs=bcs0, solver_parameters=solver_parameters)
    solve(a1 == L1, u1_sol, bcs=bcs1, solver_parameters=solver_parameters)
    solve(a2 == L2, u2_sol, bcs=bcs2, solver_parameters=solver_parameters)
    solve(a3 == L3, u3_sol, bcs=bcs3, solver_parameters=solver_parameters)
    norm_l2 = norm(u0_sol - u0_k_sol, mesh=mesh)+ norm(u1_sol - u1_k_sol, mesh=mesh) + norm(u2_sol - u2_k_sol, mesh=mesh) + norm(u3_sol - u3_k_sol, mesh=mesh)

    # Store the values
    u0_sol_vec = np.array(u0_sol.vector().dat.data)
    u0_sol_values.append(u0_sol_vec)
    u0_k_sol.assign(u0_sol)
    
    u1_sol_vec = np.array(u1_sol.vector().dat.data)
    u1_sol_values.append(u1_sol_vec)
    u1_k_sol.assign(u1_sol)
    
    u2_sol_vec = np.array(u2_sol.vector().dat.data)
    u2_sol_values.append(u2_sol_vec)
    u2_k_sol.assign(u2_sol)
    
    u3_sol_vec = np.array(u3_sol.vector().dat.data)
    u3_sol_values.append(u3_sol_vec)
    u3_k_sol.assign(u3_sol)
    
    t += dt

# *** Plotting ***
# Plot the results or save them as needed
# Colormap
fig = plt.figure(dpi=300, figsize=(8, 6))
U1 = np.array(u0_sol_values)
p = plt.imshow(U1, origin="lower", aspect='auto', cmap='jet', vmin=0.0, vmax=1.0)
clb = plt.colorbar(p)
plt.xlabel(r'x')
plt.ylabel(r't')
plt.savefig('CA_profile.png')


fig = plt.figure(dpi=300, figsize=(8, 6))
U1 = np.array(u1_sol_values)
p = plt.imshow(U1, origin="lower", aspect='auto', cmap='jet', vmin=0.0, vmax=1.0)
clb = plt.colorbar(p)
plt.xlabel(r'x')
plt.ylabel(r't')
plt.savefig('CB_profile.png')

fig = plt.figure(dpi=300, figsize=(8, 6))
U1 = np.array(u2_sol_values)
p = plt.imshow(U1, origin="lower", aspect='auto', cmap='jet', vmin=0.0, vmax=1.0)
clb = plt.colorbar(p)
plt.xlabel(r'x')
plt.ylabel(r't')
plt.savefig('CC_profile.png')

fig = plt.figure(dpi=300, figsize=(8, 6))
U1 = np.array(u3_sol_values)
p = plt.imshow(U1, origin="lower", aspect='auto', cmap='jet', vmin=0.0, vmax=1.0)
clb = plt.colorbar(p)
plt.xlabel(r'x')
plt.ylabel(r't')
plt.savefig('CD_profile.png')




## Setting up the figure object
#fig = plt.figure(dpi=300)
#ax = plt.subplot(111)
#
## Plotting the data
#for i in range(len(u0_sol_values)):
#    ax.plot(x_values, u0_sol_values[i], label=('step %i' % (i + 1)))
#
## Getting and setting the legend
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, 1.01 * box.width, box.height])
#ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#
## Setting the xy-labels
#plt.xlabel(r'$x$ [L]')
#plt.ylabel(r'$u$ [population density]')
#plt.xlim(x_values.min(), x_values.max())
#
## Setting the grids in the figure
#plt.minorticks_on()
#plt.grid(True)
#plt.grid(False, linestyle='--', linewidth=0.5, which='major')
#plt.grid(False, linestyle='--', linewidth=0.1, which='minor')
#
## Displaying the plot
#plt.show()
