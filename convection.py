"""
Script to test WNL analysis 

Usage:
    convection.py [options]

Options:
    --Lmax=<Lmax>                 Lmax resolution for simulation [default: 43]
    --Nmax=<Nmax>                 Nmax resolution for simulation [default: 48]
    --Ekman=<Ekman>               Ekman number [default: 1e-3]
    --eps=<eps>                   How far away from criticaility [default: 0.1]
"""

import numpy as np
import dedalus.public as d3
import os 
import h5py
import pandas as pd

from dedalus.extras.flow_tools import GlobalArrayReducer

from mpi4py import MPI
import time

import logging

from docopt import docopt

restart_file=False
comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size

logger = logging.getLogger(__name__)

log2 = np.log2(ncpu)

args = docopt(__doc__)

# Try to create balanced mesh                                                                                                                                                                                  
# Choose mesh whose factors are most similar in size                                                                                                                                                           
factors = [[ncpu//i,i] for i in range(1,int(np.sqrt(ncpu))+1) if np.mod(ncpu,i)==0]
score = np.array([f[1]/f[0] for f in factors])
mesh = factors[np.argmax(score)]

Lmax   = int(args['--Lmax'])
Nmax   = int(args['--Nmax'])
Ekman  = float(args['--Ekman'])
eps    = float(args['--eps'])
Prandtl = 1

file_dir = 'Ekman_{0:g}'.format(Ekman)

data = np.load('{0:s}/results.npz'.format(file_dir))

idx = np.argmin(data['Ra'])
mc = data['ms'][idx]
Rayleigh = data['Ra'][idx]*(1+eps**2)

stop_time = 100

r_inner = 7/13
r_outer = 20/13
radii = (r_inner,r_outer)

c = d3.SphericalCoordinates('phi', 'theta', 'r')

d = d3.Distributor((c,), mesh=mesh,dtype=np.float64)
b = d3.ShellBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), radii=radii, dealias=(3/2,3/2,3/2),dtype=np.float64)

s2_basis = b.S2_basis()
bk1 = b.clone_with(k=1)

phi, theta, r = b.local_grids()
phig,thetag,rg= b.global_grids()

u = d.VectorField(c,name='u',bases=b)
p = d.Field(name='p',bases=bk1)
φ = d.Field(name='φ',bases=bk1)
T = d.Field(name='T',bases=b)

tau_u1 = d.VectorField(c,name='tau_u1',bases=s2_basis)
tau_u2 = d.VectorField(c,name='tau_u2',bases=s2_basis)
tau_p = d.Field(name='tau_p')
tau_phi = d.Field(name='tau_phi')

tau_T1 = d.Field(name='tau_T1',bases=s2_basis)
tau_T2 = d.Field(name='tau_T2',bases=s2_basis)

ez = d.VectorField(c,name='ez',bases=b)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)

ro_vec =  d.VectorField(c,name='r_vec',bases=b.radial_basis)
ro_vec['g'][2] = r/r_outer

## initial condition (Load from file) ##
d_complex = d3.Distributor((c,), mesh=mesh,dtype=np.complex128)
b_complex = d3.ShellBasis(c, (2*(Lmax+1),Lmax+1,Nmax+1), radii=radii, dealias=(3/2,3/2,3/2),dtype=np.complex128)

uA = d_complex.VectorField(c,name='u',bases=b_complex)
TA = d_complex.Field(name='T',bases=b_complex)
with h5py.File('{0:s}/eigenvector/eigenvector_s1.h5'.format(file_dir), mode='r') as file:
    uA.load_from_hdf5(file,-1)
    TA.load_from_hdf5(file,-1)

uA.change_scales(1)
TA.change_scales(1)
u['g'] = uA['g'].real
T['g'] = TA['g'].real

df = pd.read_csv('{0:s}/wnl_coefficients.csv'.format(file_dir), index_col=0, header=None).T

chi   = np.complex128(df['chi'])[0]
gamma = np.complex128(df['gamma_AA'])[0] + np.complex128(df['gamma_AAbar'])[0]
predicted_amplitude = 4*eps**2*chi.real/gamma.real
logger.info('Predicted amplitude = {0:g}'.format(predicted_amplitude))

reducer = GlobalArrayReducer(MPI.COMM_WORLD)

norm = reducer.global_max((0.5*d3.integ(u@u)).evaluate()['g'])
u['g'] /= np.sqrt(norm/(0.1*predicted_amplitude))
norm = reducer.global_max((0.5*d3.integ(u@u)).evaluate()['g'])
logger.info('norm = %g' % norm.real )
########################################
rvec = d.VectorField(c, name='r_er', bases=b.radial_basis)
rvec['g'][2] = r

lift = lambda A, n: d3.Lift(A, bk1, n)

grad_u = d3.grad(u) + rvec*lift(tau_u1,-1) # First-order reduction
grad_T = d3.grad(T) + rvec*lift(tau_T1,-1) # First-order reduction

# Hydro only
problem = d3.IVP([p, u, T, tau_u1,tau_u2,tau_T1,tau_T2,tau_p], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p = 0")
problem.add_equation("dt(u) - Ekman*div(grad_u) + grad(p) + lift(tau_u2,-1) = cross(u, curl(u) + 2*ez) + Rayleigh*Ekman*ro_vec*T")
problem.add_equation("dt(T) - Ekman/Prandtl*div(grad_T) + lift(tau_T2,-1) = - dot(u,grad(T))")

problem.add_equation("integ(p)=0")
problem.add_equation("u(r=r_inner) = 0")
problem.add_equation("T(r=r_inner) = 1")
problem.add_equation("u(r=r_outer) = 0")
problem.add_equation("T(r=r_outer) = 0")

# Solver
solver = problem.build_solver(d3.SBDF2)
logger.info("Problem built")

solver.stop_sim_time = stop_time

flow = d3.GlobalFlowProperty(solver, cadence=100)
flow.add_property(d3.dot(u,u)/2., name='KE')

max_dt = 1e-4
CFL = d3.CFL(solver, initial_dt=max_dt, cadence=10, safety=0.8, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_dt)
CFL.add_velocity(u)

timeseries = solver.evaluator.add_file_handler('{0:s}/verification'.format(file_dir),sim_dt=2e-4)
timeseries.add_task('d3.integ(d3.dot(u,u))/2',name='KE')

good_solution = True

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed and good_solution:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 100 == 0:
            KE = flow.volume_integral('KE')
            good_solution = np.isfinite(KE)
            logger.info('Iteration=%i, Time=%e, dt=%e, KE=%g' %(solver.iteration, solver.sim_time, timestep, KE))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
