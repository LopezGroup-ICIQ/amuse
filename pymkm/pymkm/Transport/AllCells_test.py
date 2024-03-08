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

# Getting trial and test functions
u0, u1 = TrialFunction(V),TrialFunction(V) 
#u0_k, u1_k = TrialFunction(V),TrialFunction(V)

u0r, u1r = Function(V), Function(V)

u0_sol, u1_sol = Function(V),Function(V)
#u0_k_sol, u1_k_sol = Function(V),Function(V)

v0, v1 = TestFunction(V),TestFunction(V)


solver_parameters = {
    "ksp_type": "preonly",  # This set the method to perform only the preconditioner (LU, in the case)
    "pc_type": "lu"  # The desired preconditioner (LU)
}

# Convergence criteria
norm_l2 = 1.0e5  # Any arbitrary value greater than the tolerance
tolerance = 1.e-5

# Initial condition
x = SpatialCoordinate(mesh)
expr0 = Constant(1.0) * exp(- 1000.0 * x[0] * x[0]) #Constant(0.0) #exp(- 000.0 * x[0] * x[0])  # An expression to the initial condition
expr1 = 0.0 * x[0]
ic0 = Function(V).interpolate(expr1)
ic1 = Function(V).interpolate(expr0)




# Essential boundary conditions
boundary_value_u = 1.0
boundary_value_v = 0.0

bcs0 = DirichletBC(V, boundary_value_u, [1])  # Boundary condition in 1 and 2 marked bounds (left and right)
bcs1 = DirichletBC(V, boundary_value_v, [1,2])  # Boundary condition in 1 and 2 marked bounds (left and right)

#u0_k_sol.assign(ic0)
u0_sol.assign(ic0)

#u1_k_sol.assign(ic0)
u1_sol.assign(ic0)



dt = 0.01



u0r.assign(ic0)
u1r.assign(ic0)

def solve_function():
    
    

 
    F0 = inner((u0-u0r) / dt, v0) * dx + inner(Constant(0.01) * grad(u0), grad(v0)) * dx 
    
    
    
    # ** V part **
    F1 = inner((u1-u1r) / dt, v1) * dx + inner(Constant(0.01) * grad(u1), grad(v1)) * dx 
    
    a0, L0 = lhs(F0), rhs(F0)
    a1, L1 = lhs(F1), rhs(F1)


    solve(a0 == L0, u0_sol, bcs=bcs0, solver_parameters=solver_parameters)
    solve(a1 == L1, u1_sol, bcs=bcs1, solver_parameters=solver_parameters)

    print(u0_sol.dat.data[0])
    print(u1_sol.dat.data[0])

    ws0 = []
    ws1 = []
    for ii,jj,index in zip(u0_sol.dat.data, u1_sol.dat.data, range(len(u1_sol.dat.data))):
        if index == 0:

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
        else:
            ii = ii
            jj = jj
        
        ws0.append(ii)
        ws1.append(jj) 

    u0_sol.dat.data[:] = np.asarray(ws0)
    u1_sol.dat.data[:] = np.asarray(ws1)

    print(u0_sol.dat.data[0])
    print(u1_sol.dat.data[0])






 

    #F0 += (u-u0r) * p * dx
    #F1 += (v-u1r) * q * dx 



    #u0r.dat.data[:] = u0_sol.dat.data[:]
    #u1r.dat.data[:] = u1_sol.dat.data[:]
    norm_l2 = norm(u0_sol - u0r, mesh=mesh)+ norm(u1_sol - u1r, mesh=mesh)


    return u0_sol,u1_sol, norm_l2

Total_time = 1.0


norm_l2 = 1.0e5  # Any arbitrary value greater than the tolerance
tolerance = 1.e-5

# Iterating and solving over the time
step = 0
t = 0.0
x_values = mesh.coordinates.vector().dat.data
u_values = []
v_values = []
u_values_deg1 = []
v_values_deg1 = []
#usol_deg1 = Function(Vref)
#vsol_deg1 = Function(Vref)
while t < Total_time and norm_l2 > tolerance:
    step += 1
    print('============================')
    print('\ttime =', t)
    print('\tstep =', step)
    print('============================')

    # solve(F == 0, w, bcs=[u_bc, v_bc], solver_parameters=solver_parameters)
    #solve(F == 0, w, solver_parameters=solver_parameters)
    #w0.assign(w)

    usol, vsol, norm_l2 = solve_function()
    

    #usol_deg1.project(usol)
    #vsol_deg1.project(vsol)
    u_vec = np.array(usol.vector().dat.data)
    u_values.append(u_vec)
    #u_vec_deg1 = np.array(u_vec.vector().dat.data)
    #u_values_deg1.append(u_vec_deg1)
    v_vec = np.array(vsol.vector().dat.data)
    v_values.append(v_vec)
    #v_vec_deg1 = np.array(v_vec.vector().dat.data)
    #v_values_deg1.append(v_vec_deg1)

    t += dt


# Setting up the figure object

#for t in range(step):
#    fig = plt.figure(dpi=300, figsize=(8, 6))
#    ax = plt.subplot(111)
    # Plotting
#    ax.plot(x_values, u_values_deg1[t], '--', label='U')
#    ax.plot(x_values, v_values_deg1[t], label='V')
    # Getting and setting the legend
#    box = ax.get_position()
#    ax.set_position([box.x0, box.y0, 1.05 * box.width, box.height])
#    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Setting the xy-labels
#    plt.xlabel(r'$x$ [L]')
#    plt.ylabel(r'concentration [adim]')
#    plt.xlim(x_values.min(), x_values.max())
    
    # Setting the grids in the figure
#    plt.minorticks_on()
#    plt.grid(True)
#    plt.grid(False, linestyle='--', linewidth=0.5, which='major')
#    plt.grid(False, linestyle='--', linewidth=0.1, which='minor')
    
#    plt.tight_layout()
#    plt.savefig('gray-scott_'+str(t)+'.png')
# plt.show()

# Colormap
fig = plt.figure(dpi=300, figsize=(8, 6))
Vplot = np.array(v_values)
p = plt.imshow(Vplot, origin="lower", aspect='auto', cmap='jet')
clb = plt.colorbar(p)
plt.xlabel(r'x')
plt.ylabel(r't')
plt.savefig('gray-scott-pattern_v.png')

fig = plt.figure(dpi=300, figsize=(8, 6))
Vplot = np.array(u_values)
p = plt.imshow(Vplot, origin="lower", aspect='auto', cmap='jet')
clb = plt.colorbar(p)
plt.xlabel(r'x')
plt.ylabel(r't')
plt.savefig('gray-scott-pattern_u.png')
# plt.show()