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

solver_parameters = {
    "ksp_type": "preonly",  # This set the method to perform only the preconditioner (LU, in the case)
    "pc_type": "lu"  # The desired preconditioner (LU)
}

# Mesh definition

numel = 2500
max_x = 0
min_x = -1.0

x_left, x_right = min_x, max_x
mesh1 = IntervalMesh(numel, x_left, x_right)
    
    
# Function space declaration
degree = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh1, "CG", degree)
    
# Getting trial and test functions
u0, u1 = TrialFunction(V),TrialFunction(V) 
#u0_k, u1_k = TrialFunction(V),TrialFunction(V)
    
u0r, u1r = Function(V), Function(V)
    
u0_sol, u1_sol = Function(V),Function(V)

    
v0, v1 = TestFunction(V),TestFunction(V)
    
    

    
# Convergence criteria
norm_l2 = 1.0e5  # Any arbitrary value greater than the tolerance
tolerance = 1.e-5
    
# Initial condition
x = SpatialCoordinate(mesh1)

expr1 = 0.0 * x[0]
ic0 = Function(V).interpolate(expr1)
    
    
boundary_value_u = 1.0
boundary_value_v = 0.0
    
# Essential boundary conditions
    
bcs0 = DirichletBC(V, boundary_value_u, [1])  # Boundary condition in 1 and 2 marked bounds (left and right)
bcs1 = DirichletBC(V, boundary_value_v, [1])  # Boundary condition in 1 and 2 marked bounds (left and right)
    
#u0_k_sol.assign(ic0)
u0_sol.assign(ic0)
    
#u1_k_sol.assign(ic0)
u1_sol.assign(ic0)
    
    
    
    
    
    
u0r.assign(ic0)
u1r.assign(ic0)
    



Total_time = 1.0


norm_l2 = 1.0e5  # Any arbitrary value greater than the tolerance
tolerance = 1.e-5

dt = 0.01

# Iterating and solving over the time
step = 0
t = 0.0
u_values = []
v_values = []
u_values2 = []
v_values2 = []
while t < Total_time and norm_l2 > tolerance:
    step += 1
    print('============================')
    print('\ttime =', t)
    print('\tstep =', step)
    print('============================')

    F0 = inner((u0-u0r) / dt, v0) * dx + inner(Constant(100) * grad(u0), grad(v0)) * dx
    F1 = inner((u1-u1r) / dt, v1) * dx + inner(Constant(100) * grad(u1), grad(v1)) * dx

    a0, L0 = lhs(F0), rhs(F0)
    a1, L1 = lhs(F1), rhs(F1)

    
    solve(a0 == L0, u0_sol, bcs=bcs0, solver_parameters=solver_parameters)
    solve(a1 == L1, u1_sol, bcs=bcs1, solver_parameters=solver_parameters) 

    norm_l2 = norm(u0_sol - u0r, mesh=mesh1)+ norm(u1_sol - u1r, mesh=mesh1)
    

    u_vec = np.array(u0_sol.vector().dat.data)
    u_values.append(u_vec)
    

    v_vec = np.array(u1_sol.vector().dat.data)
    v_values.append(v_vec)

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




max_x2 = 1.0
min_x2 = 0.0

x_left2, x_right2 = min_x2, max_x2
mesh2 = IntervalMesh(numel, x_left2, x_right2)
    
    
# Function space declaration
degree = 1  # Polynomial degree of approximation
V2 = FunctionSpace(mesh2, "CG", degree)
    
# Getting trial and test functions
u02, u12 = TrialFunction(V2),TrialFunction(V2) 
#u0_k, u1_k = TrialFunction(V),TrialFunction(V)
    
u0r2, u1r2 = Function(V2), Function(V2)
    
u0_sol2, u1_sol2 = Function(V2),Function(V2)

    
v02, v12 = TestFunction(V2),TestFunction(V2)
    
    

x2 = SpatialCoordinate(mesh2)
expr12 =  0 * x2[0]
expr22 = 0 *x2[0]
# Initial condition
ic02 = Function(V2).interpolate(expr12)
ic12 = Function(V2).interpolate(expr22)
    


    
# Essential boundary conditions
    
bcs02 = DirichletBC(V2, ii, [1])  # Boundary condition in 1 and 2 marked bounds (left and right)
bcs12 = DirichletBC(V2, jj, [1])  # Boundary condition in 1 and 2 marked bounds (left and right)
    
#u0_k_sol.assign(ic0)
u0_sol2.assign(ic02)
    
#u1_k_sol.assign(ic0)
u1_sol2.assign(ic12)
    
    
    
    
    
    
u0r2.assign(ic02)
u1r2.assign(ic12)

norm_l22 = 1e5

t = 0.0

while t < Total_time and norm_l22 > tolerance:
    step += 1
    print('============================')
    print('\ttime =', t)
    print('\tstep =', step)
    print('============================')

    # solve(F == 0, w, bcs=[u_bc, v_bc], solver_parameters=solver_parameters)
    #solve(F == 0, w, solver_parameters=solver_parameters)
    #w0.assign(w)

    #usol2, vsol2, norm_l22 = solve_function2(dt, u02, v02, u12, v12, u0_sol2, u1_sol2, u0r2, u1r2, mesh2, bcs02, bcs12)
    F02 = inner((u02-u0r2) / dt, v02) * dx + inner(Constant(100) * grad(u02), grad(v02)) * dx
    F12 = inner((u12-u1r2) / dt, v12) * dx + inner(Constant(100) * grad(u12), grad(v12)) * dx

    a02, L02 = lhs(F02), rhs(F02)
    a12, L12 = lhs(F12), rhs(F12)

    
    solve(a02 == L02, u0_sol2, bcs=bcs02, solver_parameters=solver_parameters)
    solve(a12 == L12, u1_sol2, bcs=bcs12, solver_parameters=solver_parameters) 

    norm_l2 = norm(u0_sol2 - u0r2, mesh=mesh2)+ norm(u1_sol2 - u1r2, mesh=mesh2)


    u_vec = np.array(u0_sol2.vector().dat.data)
    u_values2.append(u_vec)
    

    v_vec = np.array(u1_sol2.vector().dat.data)
    v_values2.append(v_vec)

    t += dt

total_u = np.zeros([100, 2*len(u_values[0])])

for u1,u2,index in zip(u_values, u_values2, range(len(total_u))):
    total_u[index][:2501] = u1
    total_u[index][2501:] = u2


total_v = np.zeros([100, 2*len(v_values[0])])

for v1,v2,index in zip(v_values, v_values2, range(len(total_v))):
    total_v[index][:2501] = v1
    total_v[index][2501:] = v2


# Colormap
fig = plt.figure(dpi=300, figsize=(8, 6))
Vplot = total_v
p = plt.imshow(Vplot, origin="lower", aspect='auto', cmap='jet')
clb = plt.colorbar(p)
plt.xlabel(r'x')
plt.ylabel(r't')
plt.savefig('gray-scott-pattern_v_2.png')

fig = plt.figure(dpi=300, figsize=(8, 6))
Vplot = total_u
p = plt.imshow(Vplot, origin="lower", aspect='auto', cmap='jet')
clb = plt.colorbar(p)
plt.xlabel(r'x')
plt.ylabel(r't')
plt.savefig('gray-scott-pattern_u_2.png')
