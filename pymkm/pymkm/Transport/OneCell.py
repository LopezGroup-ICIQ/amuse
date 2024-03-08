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
numel = 1000
x_left, x_right = -1.0, 1.0
mesh = IntervalMesh(numel, x_left, x_right)
x = mesh.coordinates


# Function space declaration
degree = 1  # Polynomial degree of approximation
V = FunctionSpace(mesh, "CG", degree)
W = MixedFunctionSpace((V, V))
Vref = FunctionSpace(mesh, "CG", 1)

# Getting trial and test functions
w = Function(W)
u, v = split(w)
p,q = TestFunction(W)


solver_parameters = {
    'mat_type': 'aij',
    'snes_tyoe': 'newtonls',
    'pc_type': 'lu'
}

w0 = Function(W)
u0, v0 = w0.split()


u0.interpolate(Constant(1.0) * exp(- 1000.0 * x[0] * x[0]))
v0.interpolate(Constant(0.0) * exp(- 1000.0 * x[0] * x[0]))




boundary_value_u = 1.0
boundary_value_v = 0.0
u_bc = DirichletBC(W.sub(0), boundary_value_u, [1])  # Boundary condition in 1 and 2 marked bounds (left and right)
v_bc = DirichletBC(W.sub(1), boundary_value_v, [1, 2])  # Boundary condition in 1 and 2 marked bounds (left and right)



def solve_function(u,p, v, q, dt):
    
    # ** U part **
    F =  inner(grad(u), grad(p)) * dx + inner( (u-u0)/dt, p) *dx 
    # ** V part **
    
    F += inner(grad(v), grad(q)) * dx + inner( (v-v0)/dt, q)  *dx

    solve(F == 0, w, solver_parameters=solver_parameters)

    

    ws0 = []
    ws1 = []
    for ii,jj,index in zip(w.dat.data[0], w.dat.data[1], range(len(w.dat.data[0]))):
        if index == 500:
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

    w.dat.data[0][:] = np.asarray(ws0)
    w.dat.data[1][:] = np.asarray(ws1)
   

    w0.assign(w)

    
    usol, vsol = w.split()

    return usol,vsol

Total_time = 1.0

outfile = File("scott.pvd")

# Iterating and solving over the time
step = 0
t = 0.0
dt = 0.01
x_values = mesh.coordinates.vector().dat.data
u_values = []
v_values = []
u_values_deg1 = []
v_values_deg1 = []
usol_deg1 = Function(Vref)
vsol_deg1 = Function(Vref)
while t < Total_time:
    step += 1
    print('============================')
    print('\ttime =', t)
    print('\tstep =', step)
    print('============================')

    # solve(F == 0, w, bcs=[u_bc, v_bc], solver_parameters=solver_parameters)
    #solve(F == 0, w, solver_parameters=solver_parameters)
    #w0.assign(w)
    usol, vsol = solve_function(u,p,v,q, dt)

    usol_deg1.project(usol)
    vsol_deg1.project(vsol)
    u_vec = np.array(usol.vector().dat.data)
    u_values.append(u_vec)
    u_vec_deg1 = np.array(usol_deg1.vector().dat.data)
    u_values_deg1.append(u_vec_deg1)
    v_vec = np.array(vsol.vector().dat.data)
    v_values.append(v_vec)
    v_vec_deg1 = np.array(vsol_deg1.vector().dat.data)
    v_values_deg1.append(v_vec_deg1)

    t += dt


# Setting up the figure object

for t in range(step):
    fig = plt.figure(dpi=300, figsize=(8, 6))
    ax = plt.subplot(111)
    # Plotting
    ax.plot(x_values, u_values_deg1[t], '--', label='U')
    ax.plot(x_values, v_values_deg1[t], label='V')
    # Getting and setting the legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, 1.05 * box.width, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Setting the xy-labels
    plt.xlabel(r'$x$ [L]')
    plt.ylabel(r'concentration [adim]')
    plt.xlim(x_values.min(), x_values.max())
    
    # Setting the grids in the figure
    plt.minorticks_on()
    plt.grid(True)
    plt.grid(False, linestyle='--', linewidth=0.5, which='major')
    plt.grid(False, linestyle='--', linewidth=0.1, which='minor')
    
    plt.tight_layout()
    plt.savefig('gray-scott_microrx_'+str(t)+'.png')
# plt.show()

# Colormap
fig = plt.figure(dpi=300, figsize=(8, 6))
Vplot = np.array(v_values_deg1)
p = plt.imshow(Vplot, origin="lower", aspect='auto', cmap='jet')
clb = plt.colorbar(p)
plt.xlabel(r'x')
plt.ylabel(r't')
plt.savefig('gray-scott-pattern_v_microrx.png')

fig = plt.figure(dpi=300, figsize=(8, 6))
Vplot = np.array(u_values_deg1)
p = plt.imshow(Vplot, origin="lower", aspect='auto', cmap='jet')
clb = plt.colorbar(p)
plt.xlabel(r'x')
plt.ylabel(r't')
plt.savefig('gray-scott-pattern_u_microrx.png')
# plt.show()