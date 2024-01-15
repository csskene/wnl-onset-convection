"""
Script to find the weakly non-linear coefficients

Usage:
    wnl_coefficients.py [options]

Options:
    --Lmax=<Lmax>                 Lmax resolution for simulation [default: 43]
    --Nmax=<Nmax>                 Nmax resolution for simulation [default: 48]
    --Ekman=<Ekman>               Ekman number [default: 1e-3]
    --Prandtl=<Prandtl>           Prandtl number [default: 1]
    --beta=<beta>                 Radius ratio [default: 0.35]
"""

import numpy as np
import dedalus.public as d3

from mpi4py import MPI
import time
import h5py
import logging
import dedalus.core.evaluator as evaluator
from docopt import docopt
import pandas as pd

restart_file=False
comm = MPI.COMM_WORLD
rank = comm.rank
ncpu = comm.size

logger = logging.getLogger(__name__)


args = docopt(__doc__)
# Parameters
Lmax             = int(args['--Lmax'])
Nmax             = int(args['--Nmax'])
Ekman            = float(args['--Ekman'])
Prandtl          = float(args['--Prandtl'])

beta =  float(args['--beta'])

r_outer = 1/(1-beta)
r_inner = r_outer - 1

# file_dir = 'Ekman_{0:g}'.format(Ekman)
file_dir = 'Ekman_{0:g}_Prandtl_{1:g}_beta_{2:g}'.format(Ekman,Prandtl,beta)

data = np.load('{0:s}/results.npz'.format(file_dir))

idx = np.argmin(data['Ra'])
mc = data['ms'][idx]
Rayleigh_ = data['Ra'][idx]
om_A = np.squeeze(data['eigs'][idx]).imag
logger.info('Calculating coefficients for mc={0:d}, Ra={1:f}, om_A={2:f}'.format(mc,Rayleigh_,om_A))

# Rayleigh_ = 55.904298660837235
# mc = 4
# target = -0.023092j


# Ekman = 1e-4
# Rayleigh_ = 69.53045795368095
# mc = 7 
# target = -0.01332796j

# Lmax = 183
# Nmax = 183
# Ekman = 1e-5
# Rayleigh_ = 105.567
# mc = 15
# target = -0.0071233j

r_inner = 7/13
r_outer = 20/13
radii = (r_inner,r_outer)

vol = 4*np.pi/3*(r_outer**3-r_inner**3)

c = d3.SphericalCoordinates('phi', 'theta', 'r')
d = d3.Distributor((c,), dtype=np.complex128,comm=MPI.COMM_SELF)
b = d3.ShellBasis(c, (2*(2*mc+1),Lmax+1,Nmax+1), radii=radii, dealias=(3/2,3/2,3/2),dtype=np.complex128)
s2_basis = b.S2_basis()

b_inner = b.S2_basis(radius=r_inner)
b_outer = b.S2_basis(radius=r_outer)
phi, theta, r = d.local_grids(b)

u = d.VectorField(c,name='u',bases=b)
p = d.Field(name='p',bases=b)
φ = d.Field(name='φ',bases=b)
T = d.Field(name='T',bases=b)
T0 = d.Field(name='T',bases=b.meridional_basis)

tau_u1 = d.VectorField(c,name='tau_u1',bases=s2_basis)
tau_u2 = d.VectorField(c,name='tau_u2',bases=s2_basis)
tau_p = d.Field(name='tau_p')
tau_phi = d.Field(name='tau_phi')

Rayleigh = d.Field(name='Rayleigh')
Rayleigh['g'] = Rayleigh_
tau_T1 = d.Field(name='tau_T1',bases=s2_basis)
tau_T2 = d.Field(name='tau_T2',bases=s2_basis)

ez = d.VectorField(c,name='ez',bases=b.meridional_basis)
ez['g'][1] = -np.sin(theta)
ez['g'][2] =  np.cos(theta)

r_vec =  d.VectorField(c,name='r_vec',bases=b.meridional_basis)
r_vec['g'][2] = r/r_outer

T_inner = d.Field(name='T_inner',bases=b.S2_basis(r_inner))
T_inner['g'] = 1.

# initial condition
amp = 0.1
x = 2*r-r_inner-r_outer
T0['g'] = r_inner*r_outer/r - r_inner

rvec = d.VectorField(c, name='er', bases=b.meridional_basis)
rvec['g'][2] = r

lift_basis = b.clone_with(k=1) # First derivative basis
lift = lambda A, n: d3.Lift(A, lift_basis, n)
integ = lambda A: d3.Integrate(A, c)
grad_u = d3.grad(u) + rvec*lift(tau_u1,-1) # First-order reduction
grad_T = d3.grad(T) + rvec*lift(tau_T1,-1) # First-order reduction

with h5py.File('{0:s}/eigenvector/eigenvector_s1.h5'.format(file_dir), mode='r') as file:
    for state_variable in [u,T]:
        state_variable.load_from_hdf5(file,-1)
uA = u.copy()
TA = T.copy()

ureal = uA.copy()
ureal['g'] = uA['g'].real
norm = (0.5*d3.integ(ureal@ureal)).evaluate()['g'][0,0,0]
logger.info('norm = %f' % norm.real )

## Check energy balance
D_nu = np.max(d3.integ(d3.dot(ureal,d3.lap(ureal))).evaluate()['g'])
strain = 0.5*(d3.grad(ureal) + d3.trans(d3.grad(ureal)))
De = 2*np.max(d3.integ(d3.trace(strain@strain)).evaluate()['g'])
logger.info('Residual = {0:g}'.format(np.abs(D_nu+De)/np.max([np.abs(De),np.abs(D_nu)])))

uAbar = uA.copy()
uAbar['g'] = np.conj(uA['g'])

TAbar = TA.copy()
TAbar['g'] = np.conj(TA['g'])

with h5py.File('{0:s}/eigenvector_adj/eigenvector_adj_s1.h5'.format(file_dir), mode='r') as file:
    for state_variable in [u,T]:
        state_variable.load_from_hdf5(file,0)
uA_adj = u.copy()
TA_adj = T.copy()

term = 0
term += np.vdot(uA_adj['c'],uA['c'])
term += np.vdot(TA_adj['c'],TA['c'])

logger.info('Check adjoint normalisation = {0:f}'.format(term))

########################
## Second order terms ##
########################

############
## uAAbar ##
############

problem = d3.LBVP([p, u, T, tau_u1,tau_u2,tau_T1,tau_T2,tau_p], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p= 0")
problem.add_equation("-Ekman*div(grad_u) + grad(p) + lift(tau_u2,-1) - Rayleigh*Ekman*r_vec*T + 2*cross(ez, u) = -uAbar@d3.grad(uA) - uA@d3.grad(uAbar)")
problem.add_equation("-div(grad_T)*Ekman/Prandtl + lift(tau_T2,-1) + dot(u,grad(T0)) = -uAbar@d3.grad(TA) - uA@d3.grad(TAbar)")

problem.add_equation("integ(p)=0")
problem.add_equation("u(r=r_inner) = 0")
problem.add_equation("T(r=r_inner) = 0")
problem.add_equation("u(r=r_outer) = 0")
problem.add_equation("T(r=r_outer) = 0")

solver = problem.build_solver(ncc_cutoff=1e-6)

logger.info('Solving for uAAbar')
u['g'] = 0
T['g'] = 0

subproblem = solver.subproblems_by_group[(0, None, None)]
solver.solve(subproblem)

uAAbar = u.copy()
TAAbar = T.copy()

#########
## uAA ##
#########
# m = 2m_c term
# uAA
logger.info('Solving for uAA')
u['g'] = 0
T['g'] = 0
problem = d3.LBVP([p, u, T, tau_u1,tau_u2,tau_T1,tau_T2,tau_p], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p= 0")
problem.add_equation("2j*om_A*u - Ekman*div(grad_u) + grad(p) + lift(tau_u2,-1) - Rayleigh*Ekman*r_vec*T + 2*cross(ez, u) = -uA@grad(uA)")
problem.add_equation("2j*om_A*T - div(grad_T)*Ekman/Prandtl + lift(tau_T2,-1) + dot(u,grad(T0)) = -uA@grad(TA)")

problem.add_equation("integ(p)=0")
problem.add_equation("u(r=r_inner) = 0")
problem.add_equation("T(r=r_inner) = 0")
problem.add_equation("u(r=r_outer) = 0")
problem.add_equation("T(r=r_outer) = 0")

solver = problem.build_solver(ncc_cutoff=1e-6)

subproblem = solver.subproblems_by_group[(2*mc, None, None)]

solver.solve(subproblem)

uAA = u.copy()
TAA = T.copy()

##############################
## Compute WNL coefficients ##
##############################

logger.info('Computing WNL terms')

gamma_u = u.copy()
gamma_T = T.copy()
gamma_u.change_scales(3/2)
gamma_T.change_scales(3/2)
gamma_u['g'] = (uAA@d3.grad(uAbar) + uAbar@d3.grad(uAA) + uA@d3.grad(uAAbar) + uAAbar@d3.grad(uA)).evaluate()['g']
gamma_T['g'] = (uAA@d3.grad(TAbar) + uAbar@d3.grad(TAA) + uA@d3.grad(TAAbar) + uAAbar@d3.grad(TA)).evaluate()['g']
gamma = np.vdot(uA_adj['c'],gamma_u['c']) + np.vdot(TA_adj['c'],gamma_T['c'])
logger.info('gamma={0:g}'.format(gamma))

gamma_u_AA = u.copy()
gamma_T_AA = T.copy()
gamma_u_AA['g'] = (uAA@d3.grad(uAbar) + uAbar@d3.grad(uAA)).evaluate()['g']
gamma_T_AA['g'] = (uAA@d3.grad(TAbar) + uAbar@d3.grad(TAA)).evaluate()['g']
gamma_AA_u = np.vdot(uA_adj['c'],gamma_u_AA['c'])
gamma_AA_T = np.vdot(TA_adj['c'],gamma_T_AA['c'])
gamma_AA = gamma_AA_u + gamma_AA_T
logger.info('gamma_AA={0:g}'.format(gamma_AA))

gamma_u_AAbar = u.copy()
gamma_T_AAbar = T.copy()
gamma_u_AAbar['g'] = (uA@d3.grad(uAAbar) + uAAbar@d3.grad(uA)).evaluate()['g']
gamma_T_AAbar['g'] = (uA@d3.grad(TAAbar) + uAAbar@d3.grad(TA)).evaluate()['g']
gamma_AAbar_u = np.vdot(uA_adj['c'],gamma_u_AAbar['c'])
gamma_AAbar_T = np.vdot(TA_adj['c'],gamma_T_AAbar['c'])
gamma_AAbar = gamma_AAbar_u + gamma_AAbar_T
logger.info('gamma_AAbar={0:g}'.format(gamma_AAbar))

assert(np.allclose(gamma_AA+gamma_AAbar,gamma))

chi_u = u.copy()
chi_T = T.copy()
chi_u.change_scales(3/2)
chi_T.change_scales(3/2)

chi_u['g'] = (Rayleigh*Ekman*r_vec*TA).evaluate()['g']
chi = np.vdot(uA_adj['c'],chi_u['c'])
logger.info('chi={0:g}'.format(chi))

logger.info('Amplitude={0:g}'.format(np.sqrt(chi.real/gamma.real)))

################
## Save terms ##
################

output_evaluator = evaluator.Evaluator(d, locals())
output_handler = output_evaluator.add_file_handler('{0:s}/wnl_terms'.format(file_dir))

output_handler.add_task(uAA,name='uAA')
output_handler.add_task(uAA,name='TAA')

output_handler.add_task(uAAbar,name='uAAbar')
output_handler.add_task(uAAbar,name='TAAbar')

output_handler.add_task(gamma_u,name='gamma_u')
output_handler.add_task(gamma_T,name='gamma_T')
output_handler.add_task(gamma_u_AA,name='gamma_u_AA')
output_handler.add_task(gamma_T_AA,name='gamma_T_AA')
output_handler.add_task(gamma_u_AAbar,name='gamma_u_AAbar')
output_handler.add_task(gamma_T_AAbar,name='gamma_T_AAbar')

output_handler.add_task(chi_u,name='chi_u')
output_handler.add_task(chi_T,name='chi_T')
output_evaluator.evaluate_handlers(output_evaluator.handlers, timestep=0, wall_time=0, sim_time=0, iteration=0)
#, index=[1e-3],columns=['gamma_AA','gamma_AA_u','gamma_AA_T','gamma_AAbar','gamma_AAbar_u','gamma_AAbar_T','chi']
df = pd.DataFrame(np.array([gamma_AA,gamma_AA_u,gamma_AA_T,gamma_AAbar,gamma_AAbar_u,gamma_AAbar_T,chi]),index=['gamma_AA','gamma_AA_u','gamma_AA_T','gamma_AAbar','gamma_AAbar_u','gamma_AAbar_T','chi'],columns=[Ekman])
print(df)
df.to_csv('{0:s}/wnl_coefficients.csv'.format(file_dir))