from firedrake import *
from mkm import MKM

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
plt.rc('text', usetex=True)
plt.rc('font', size=14)

T = 473
P = 1e6

model = MKM('Test'+str(T), './rm_transport.mkm', 
            './g_transport.mkm', t_ref=T)

model.set_reactor('dynamic')
model.set_CSTR_params(radius=0.0025,
                      length=0.0050,
                      Q=0.66E-6,
                      S_BET=1.74E5,
                      m_cat=1.0E-4)

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
D0 = Constant(1) #Constant(1.e0)
D1 = Constant(1)
D2 = Constant(0.01)
D3 = Constant(0.01)
K = Constant(0.0)



# Source term
#f = Constant(0.0)
f0 = -K * u0_sol * u1_sol
f1 = -K * u0_sol * u1_sol
f2 = K * u0_sol * u1_sol
f3 = Constant(0.0)

# Advection
r = Constant(0.0)


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
bcs1 = DirichletBC(V, boundary_value, [1])  # Boundary condition in 1 and 2 marked bounds (left and right)
bcs2 = DirichletBC(V, boundary_value, [1])  # Boundary condition in 1 and 2 marked bounds (left and right)
bcs3 = DirichletBC(V, boundary_value, [1])  # Boundary condition in 1 and 2 marked bounds (left and right)


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
t = 0.0
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

    #u0r, u1r = g(u0_sol, u1_sol, 2500)
    #print(u0r.dat.data[2500])
    #print(u1r.dat.data[2500])

    #F0 += (u0-u0r) * v0 * ds(2) + Constant(0.0) * v0 * ds(1)
    
    

    #F1 += (u1-u1r) * v1 * ds(2) + Constant(0.0) * v1 * ds(1)
    

    



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


print("The reactive wall is reached")

ii = u0_sol.dat.data[-1]
jj = u1_sol.dat.data[-1]

if ii < 1e-5:
    ii = ii
    jj = jj
else:
    if jj + ii != 1.0:
        ii = round(ii,2)
        jj = round(jj,2)
        try:
            x = model.kinetic_run(T, P, np.asarray([ii,jj]))
        except:
            new_ii = ii / (ii + jj)
            new_jj = 1 - new_ii
            try:
                x = model.kinetic_run(T, P, np.asarray([new_ii,new_jj]))
            except:
                ii = ii
                jj = jj
            else:
                ii = x["y_out"]["A(g)"] * (ii+jj)
                jj = x["y_out"]["B(g)"] * (ii+jj) 
        else:
            ii = x["y_out"]["A(g)"]
            jj = x["y_out"]["B(g)"]
    elif jj + ii == 1.0:
        try:
            x = model.kinetic_run(T, P, np.asarray([ii,jj]))
        except:
            ii = ii
            jj = jj
        else:
            ii = x["y_out"]["A(g)"]
            jj = x["y_out"]["B(g)"]
    else:
        ii = ii
        jj = jj
    

print("The reaction took place and the value of the fields are: u="+str(ii)+" v="+str(jj))


u0_sol_values[-1][-1] = ii
u1_sol_values[-1][-1] = jj




# Colormap
fig = plt.figure(dpi=300, figsize=(8, 6))
Vplot = np.array(u0_sol_values)
p = plt.imshow(Vplot, origin="lower", aspect='auto', cmap='jet')
clb = plt.colorbar(p)
plt.xlabel(r'x')
plt.ylabel(r't')
plt.savefig('gray-scott-pattern_u_2_end.png')

fig = plt.figure(dpi=300, figsize=(8, 6))
Vplot = np.array(u1_sol_values)
p = plt.imshow(Vplot, origin="lower", aspect='auto', cmap='jet')
clb = plt.colorbar(p)
plt.xlabel(r'x')
plt.ylabel(r't')
plt.savefig('gray-scott-pattern_v_2_end.png')
