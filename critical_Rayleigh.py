"""
Script to find the critical Rayleigh number for different m's

Usage:
    critical_Rayleigh.py [options]

Options:
    --Lmax=<Lmax>                 Lmax resolution for simulation [default: 43]
    --Nmax=<Nmax>                 Nmax resolution for simulation [default: 48]
    --Ekman=<Ekman>               Ekman number [default: 1e-3]
    --Rayleigh=<Rayleigh>         Order of magnitude guess for Rayleigh number [default: 45]
    --m_min=<m_min>               Starting m [default: 4]
    --m_max=<m_max>               Final m [default: 4]
    --target=<target>             Target frequency [default: -0.2]
    --Prandtl=<Prandtl>           Prandtl number [default: 1]
    --beta=<beta>                 Radius ratio [default: 0.35]
    --internal                    Whether to use internal heating or not
    --recalculate                 Just recalculate the eigenvector
"""

import numpy as np
import dedalus.public as d3
import os
import time
import tools
import logging
import scipy
import copy
from docopt import docopt
from scipy.optimize import minimize
import dedalus.core.evaluator as evaluator

logger = logging.getLogger(__name__)

args = docopt(__doc__)

# parameters
Lmax             = int(args['--Lmax'])
Nmax             = int(args['--Nmax'])
Ekman            = float(args['--Ekman'])
Rayleigh_        = float(args['--Rayleigh'])

m_min        = int(args['--m_min'])
m_max        = int(args['--m_max'])
target       = float(args['--target'])*1j

Prandtl =  float(args['--Prandtl'])
beta    =  float(args['--beta'])

internal = args['--internal']
recalculate = args['--recalculate']

r_outer = 1/(1-beta)
r_inner = r_outer - 1
radii = (r_inner,r_outer)

vol = 4*np.pi/3*(r_outer**3-r_inner**3)

if not os.path.exists('data'):
    os.mkdir('data')

# Create output directory
file_dir = 'data/Ekman_{0:g}_Prandtl_{1:g}_beta_{2:g}_internal_{3:s}'.format(Ekman, Prandtl, beta, str(internal))
if not os.path.exists(file_dir):
    os.mkdir(file_dir)

# Coordinates and bases
c = d3.SphericalCoordinates('phi', 'theta', 'r')
d = d3.Distributor((c,), dtype=np.complex128)
b = d3.ShellBasis(c, (2*m_max+2, Lmax+1, Nmax+1), radii=radii, dealias=(1,1,1), dtype=np.complex128)
s2_basis = b.S2_basis()

b_inner = b.S2_basis(radius=r_inner)
b_outer = b.S2_basis(radius=r_outer)
phi, theta, r = d.local_grids(b)

# Fields
u = d.VectorField(c, name='u', bases=b)
p = d.Field(name='p', bases=b)
T = d.Field(name='T', bases=b)
T0 = d.Field(name='T0', bases=b.meridional_basis)

tau_u1 = d.VectorField(c,name='tau_u1', bases=s2_basis)
tau_u2 = d.VectorField(c,name='tau_u2', bases=s2_basis)
tau_p = d.Field(name='tau_p')
tau_phi = d.Field(name='tau_phi')

eig_save = d.Field(name='eig_save')
Rayleigh = d.Field(name='Rayleigh')
Rayleigh['g'] = Rayleigh_
tau_T1 = d.Field(name='tau_T1', bases=s2_basis)
tau_T2 = d.Field(name='tau_T2', bases=s2_basis)

ez = d.VectorField(c,name='ez', bases=b.meridional_basis)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)

r_vec =  d.VectorField(c,name='r_vec', bases=b.meridional_basis)
r_vec['g'][2] = r/r_outer

# initial condition
if internal:
    logger.info("Using internal heating")
    T0['g'] = -(1-beta)/(1+beta)*r**2 + (1-beta)/(1+beta)*r_outer**2
else:
    logger.info("Using differential heating")
    T0['g'] = r_inner*r_outer/r - r_inner

rvec = d.VectorField(c, name='er', bases=b.meridional_basis)
rvec['g'][2] = r

# lifting
lift_basis = b.clone_with(k=1) # First derivative basis
lift = lambda A, n: d3.Lift(A, lift_basis, n)

lift_basis2 = b.clone_with(k=2) # Second derivative basis                                                                                                         
lift2 = lambda A, n: d3.Lift(A, lift_basis2, n)

integ = lambda A: d3.Integrate(A, c)
grad_u = d3.grad(u) + rvec*lift(tau_u1,-1) # First-order reduction
grad_T = d3.grad(T) + rvec*lift(tau_T1,-1) # First-order reduction

# Eigenvalue problem
om = d.Field(name='om')
problem = d3.EVP([p, u, T, tau_u1, tau_u2, tau_T1, tau_T2, tau_p], eigenvalue=om, namespace=locals())
problem.add_equation("trace(grad_u) + tau_p= 0")
problem.add_equation("om*u - Ekman*div(grad_u) + grad(p) + lift2(tau_u2,-1) - Rayleigh*Ekman*r_vec*T + 2*cross(ez, u) =  0")
problem.add_equation("om*T - Ekman/Prandtl*div(grad_T) + lift2(tau_T2,-1) + dot(u,grad(T0)) = 0")
problem.add_equation("integ(p)=0")
problem.add_equation("u(r=r_inner) = 0")
problem.add_equation("T(r=r_inner) = 0")
problem.add_equation("u(r=r_outer) = 0")
problem.add_equation("T(r=r_outer) = 0")
solver = problem.build_solver(ncc_cutoff=1e-10)

# For the sensitivity
dLdRa = problem.eqs[1]['L'].sym_diff(Rayleigh)

# Function to return cost and gradient for optimisation problem
def cost_grad(Rayleigh_,solver,m,target):
    # Solver
    Rayleigh['g'] = Rayleigh_[0]

    subproblem = solver.subproblems_by_group[(m, None, None)]

    nev = 5
    solver.solve_sparse(subproblem, nev, target,left=True, raise_on_mismatch=False, rebuild_matrices=True)
    tools._normalize_left_eigenvectors(solver)
    idx = np.argmax(solver.eigenvalues.real)
    tools.set_state_adjoint(solver, idx, subproblem.subsystems[0])

    state_adjoint = []
    for state in solver.state:
        state_adjoint.append(state.copy())

    solver.set_state(idx, subproblem.subsystems[0])

    norm = 0
    for (i,adj_state) in enumerate(state_adjoint):
        norm += np.vdot(adj_state['c'],solver.state[i]['c'])
    logger.info('This should be 1 %g' % norm)

    for adj_state in state_adjoint:
        adj_state['c'] /= np.conj(norm)
        
    dlambdadRa = np.vdot(state_adjoint[1]['c'], -dLdRa['c'])
    
    cost = solver.eigenvalues[idx].real**2
    grad_ = 2*solver.eigenvalues[idx].real*dlambdadRa.real
    
    logger.info('Ra = {0:g}, cost = {1:g}, grad = {2:g}, growth rate = {3:g}, freq={4:g}'.format(Rayleigh_[0], cost, grad_, solver.eigenvalues[idx].real, solver.eigenvalues[idx].imag))

    eig_save['g'] = solver.eigenvalues[idx]
    return cost, grad_

if not recalculate:
    # Calculate the critical Rayleigh number for the specified m range
    Ra_cs = []
    ms    = []
    eigs  = []
    for m in range(m_min, m_max+1):
        logger.info('Looking for m={0:d}'.format(m))
        growth = -1
        # Loop to find positive growth rate
        while(growth<0):
            Rayleigh['g'] = Rayleigh_ 
            
            subproblem = solver.subproblems_by_group[(m, None, None)]
            nev = 40
            solver.solve_sparse(subproblem, nev, target, rebuild_matrices=True)
            idx = np.argmax(solver.eigenvalues.real)
            growth = solver.eigenvalues[idx].real
            freq = solver.eigenvalues[idx].imag
            target = solver.eigenvalues[idx]
            logger.info('Ra ={0:g}, Growth= {1:g}, freq = {2:g}'.format(Rayleigh_, growth,freq))
            Rayleigh_ *= 1.1
        Rayleigh_ /= 1.1
        opts = {'disp': True}

        # Now optimise to find critical Rayleigh
        sol = scipy.optimize.minimize(lambda A: cost_grad(A,solver,m,target), x0 = np.array(Rayleigh_), jac=True, method='L-BFGS-B', tol=1e-28, options=opts)
        logger.info('Ra_c = {0:g}'.format(sol.x[0]))
        logger.info('Number of function evaluations = {0:d}'.format(sol.nfev))

        Ra_cs.append(sol.x[0])
        ms.append(m)
        eigs.append(copy.copy(eig_save['g']))

        Rayleigh_ = sol.x[0]
        target = copy.copy(eig_save['g'][0,0,0])

        np.savez('{0:s}/results'.format(file_dir), ms=ms, Ra=Ra_cs, eigs=eigs)
        flg_saved = True
else:
    logger.info('Recalculating the critical eigenvector from previous run')
    try:
        saved_data = np.load('{0:s}/results.npz'.format(file_dir))
        Ra_cs = saved_data['Ra']
        ms    = saved_data['ms']
        eigs  = saved_data['eigs']
        flg_saved = True
    except:
        logger.info('No previous run detected, using input parameters')
        Ra_cs = [Rayleigh_]
        ms = [m_min]
        eig_save['g'] = target
        eigs = [eig_save['g']]
        flg_saved = False

# Recalculate critical eigenvector for lowest m
idx = np.argmin(Ra_cs)

Rayleigh['g'] = Ra_cs[idx]        
mc = ms[idx]
target = eigs[idx][0,0,0]

subproblem = solver.subproblems_by_group[(mc, None, None)]
nev = 1

logger.info('Saving critical eigenvalue mc={0:d}'.format(mc))
solver.solve_sparse(subproblem, nev, target,left=True, raise_on_mismatch=True, rebuild_matrices=True)

# Normalise and save eigenvectors
# Save memory by storing eigenvector using a meridional basis
um = d.VectorField(c,name='um',bases=b.meridional_basis)
Tm = d.Field(name='Tm',bases=b.meridional_basis)
idx = 0

solver.set_state(idx,subproblem.subsystems[0])
um['g'][:,0,:,:] = u['g'][:,0,:,:]

umreal = um.copy()
umreal['g'] = um['g'].real

umimag = um.copy()
umimag['g'] = um['g'].imag

norm = (0.5*0.5*d3.integ(umreal@umreal + umimag@umimag)).evaluate()['g'][0,0,0]
solver.eigenvectors /= np.sqrt(norm)

solver.set_state(idx,subproblem.subsystems[0])
um['g'][:,0,:,:] = u['g'][:,0,:,:]
Tm['g'][0,:,:] = T['g'][0,:,:]

umreal['g'] = um['g'].real
umimag['g'] = um['g'].imag
norm = (0.5*0.5*d3.integ(umreal@umreal + umimag@umimag)).evaluate()['g'][0,0,0]

logger.info('Norm={0:f}'.format(norm.real))

output_evaluator = evaluator.Evaluator(d, locals())
output_handler = output_evaluator.add_file_handler('{0:s}/eigenvector'.format(file_dir))
output_handler.add_task(um,name='um')
output_handler.add_task(Tm,name='Tm')
output_evaluator.evaluate_handlers(output_evaluator.handlers, timestep=0, wall_time=0, sim_time=0, iteration=0)

tools._normalize_left_eigenvectors(solver)
tools.set_state_adjoint(solver,idx,subproblem.subsystems[0])
um['g'][:,0,:,:] = u['g'][:,0,:,:]
Tm['g'][0,:,:] = T['g'][0,:,:]

output_evaluator_adj = evaluator.Evaluator(d, locals())
output_handler_adj = output_evaluator_adj.add_file_handler('{0:s}/eigenvector_adj'.format(file_dir))
output_handler_adj.add_task(um,name='um')
output_handler_adj.add_task(Tm,name='Tm')
output_evaluator_adj.evaluate_handlers(output_evaluator_adj.handlers, timestep=0, wall_time=0, sim_time=0, iteration=0)

if not flg_saved:
    eig_save['g'] = solver.eigenvalues[0] 
    np.savez('{0:s}/results'.format(file_dir),ms=ms,Ra=Ra_cs,eigs = [eig_save['g']])
