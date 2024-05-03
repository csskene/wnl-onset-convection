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

file_dir = 'Ekman_{0:g}_Prandtl_{1:g}_beta_{2:g}'.format(Ekman,Prandtl,beta)

data = np.load('{0:s}/results.npz'.format(file_dir))

idx = np.argmin(data['Ra'])
mc = data['ms'][idx]
Rayleigh_ = data['Ra'][idx]
om_A = np.squeeze(data['eigs'][idx]).imag
logger.info('Calculating coefficients for mc={0:d}, Ra={1:f}, om_A={2:f}'.format(mc,Rayleigh_,om_A))

radii = (r_inner,r_outer)

vol = 4*np.pi/3*(r_outer**3-r_inner**3)

c = d3.SphericalCoordinates('phi', 'theta', 'r')
d = d3.Distributor((c,), dtype=np.complex128,comm=MPI.COMM_SELF)
b = d3.ShellBasis(c, (2*(2*mc+2),Lmax+1,Nmax+1), radii=radii, dealias=(1,1,1),dtype=np.complex128)
s2_basis = b.S2_basis()

b_inner = b.S2_basis(radius=r_inner)
b_outer = b.S2_basis(radius=r_outer)
phi, theta, r = d.local_grids(b)

u = d.VectorField(c,name='u',bases=b)
p = d.Field(name='p',bases=b)
T = d.Field(name='T',bases=b)
T0 = d.Field(name='T0',bases=b.meridional_basis)

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

um = d.VectorField(c,name='um',bases=b.meridional_basis)
Tm = d.Field(name='Tm',bases=b.meridional_basis)

with h5py.File('{0:s}/eigenvector/eigenvector_s1.h5'.format(file_dir), mode='r') as file:
    for state_variable in [um,Tm]:
        state_variable.load_from_hdf5(file,-1)

u['g'] = um['g']*np.exp(1j*mc*phi)
T['g'] = Tm['g']*np.exp(1j*mc*phi)
uA = u.copy()
TA = T.copy()
uAm = um.copy()
TAm = Tm.copy()

ureal = uA.copy()
ureal['g'] = uA['g'].real
norm = (0.5*d3.integ(ureal@ureal)).evaluate()['g'][0,0,0]
logger.info('norm = %f' % norm.real)

## Check energy balance
D_nu = np.max(d3.integ(d3.dot(ureal,d3.lap(ureal))).evaluate()['g'])
strain = 0.5*(d3.grad(ureal) + d3.trans(d3.grad(ureal)))
De = 2*np.max(d3.integ(d3.trace(strain@strain)).evaluate()['g'])
logger.info('Residual = {0:g}'.format(np.abs(D_nu+De)/np.max([np.abs(De),np.abs(D_nu)])))

uAbar = uAm.copy()
uAbar['g'] = np.conj(uAm['g'])

TAbar = TAm.copy()
TAbar['g'] = np.conj(TAm['g'])

with h5py.File('{0:s}/eigenvector_adj/eigenvector_adj_s1.h5'.format(file_dir), mode='r') as file:
    for state_variable in [um,Tm]:
        state_variable.load_from_hdf5(file,-1)

u['g'] = um['g']*np.exp(1j*mc*phi)
T['g'] = Tm['g']*np.exp(1j*mc*phi)
uA_adj = u.copy()
TA_adj = T.copy()
uAm_adj = um.copy()
TAm_adj = Tm.copy()

term = 0
term += np.vdot(uA_adj['c'],uA['c'])
term += np.vdot(TA_adj['c'],TA['c'])

logger.info('Check adjoint normalisation = {0:f}'.format(term))

m_field = d.Field(name='m_field',bases=b.meridional_basis)
m_field['g'] = 1/(r*np.sin(theta))
#m_field = d.VectorField(c,name='m_field',bases=b)
#m_field['g'][0] = 1/(r*np.sin(theta))

#m_vec_field = d.TensorField((c,c),name='m_vec_field',bases=b)
#m_vec_field['g'][0,0] = 1/(r*np.sin(theta))
#m_vec_field['g'][1,1] = 1/(r*np.sin(theta))
#m_vec_field['g'][2,2] = 1/(r*np.sin(theta))

e_phi = d.VectorField(c,name='e_phi')
e_phi['g'][0]=1

NL_u_field = d.VectorField(c,name='NL_u_field',bases=b)
NL_T_field = d.Field(name='NL_T_field',bases=b)
def NLTerm(u1,u2,T1,T2,m1,m2):
    NL_u = -u1@d3.grad(u2) - u2@d3.grad(u1)
    NL_u += -(e_phi@u1)*(1j*m2*m_field*u2)
    if m1 != 0:
        NL_u += -(e_phi@u2)*(1j*m1*m_field*u1)

    NL_T = -u2@d3.grad(T1) - u1@d3.grad(T2)
    NL_T += -u1@(1j*m2*m_field*T2*e_phi)
    if m1 != 0:
        NL_T += -u2@(1j*m1*m_field*T1*e_phi)
    
    NL_u_field['g'] = NL_u['g']*np.exp(1j*(m1+m2)*phi)
    NL_T_field['g'] = NL_T['g']*np.exp(1j*(m1+m2)*phi)

    return 0
    
########################
## Second order terms ##
########################

############
## uAAbar ##
############
NLTerm(uAbar,uAm,TAbar,TAm,-mc,mc)
problem = d3.LBVP([p, u, T, tau_u1,tau_u2,tau_T1,tau_T2,tau_p], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p= 0")
problem.add_equation("-Ekman*div(grad_u) + grad(p) + lift(tau_u2,-1) - Rayleigh*Ekman*r_vec*T + 2*cross(ez, u) = NL_u_field")
problem.add_equation("-div(grad_T)*Ekman/Prandtl + lift(tau_T2,-1) + dot(u,grad(T0)) = NL_T_field")

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

uAAbar = d.VectorField(c,name='uAAbar',bases=b.meridional_basis)
TAAbar = d.Field(name='TAAbar',bases=b.meridional_basis)
uAAbar['g'][:,0,:,:] = u['g'][:,0,:,:]
TAAbar['g'][0,:,:] = T['g'][0,:,:]

#########
## uAA ##
#########
# m = 2m_c term
# uAA
logger.info('Solving for uAA')
u['g'] = 0
T['g'] = 0

NLTerm(uAm,uAm,TAm,TAm,mc,mc)
problem = d3.LBVP([p, u, T, tau_u1,tau_u2,tau_T1,tau_T2,tau_p], namespace=locals())
problem.add_equation("trace(grad_u) + tau_p= 0")
problem.add_equation("2j*om_A*u - Ekman*div(grad_u) + grad(p) + lift(tau_u2,-1) - Rayleigh*Ekman*r_vec*T + 2*cross(ez, u) = NL_u_field/2")
problem.add_equation("2j*om_A*T - div(grad_T)*Ekman/Prandtl + lift(tau_T2,-1) + dot(u,grad(T0)) = NL_T_field/2")

problem.add_equation("integ(p)=0")
problem.add_equation("u(r=r_inner) = 0")
problem.add_equation("T(r=r_inner) = 0")
problem.add_equation("u(r=r_outer) = 0")
problem.add_equation("T(r=r_outer) = 0")

solver = problem.build_solver(ncc_cutoff=1e-6)

subproblem = solver.subproblems_by_group[(2*mc, None, None)]

solver.solve(subproblem)
uAA = d.VectorField(c,name='uAA',bases=b.meridional_basis)
TAA = d.Field(name='TAA',bases=b.meridional_basis)
uAA['g'][:,0,:,:] = u['g'][:,0,:,:]
TAA['g'][0,:,:] = T['g'][0,:,:]

##############################
## Compute WNL coefficients ##
##############################

logger.info('Computing WNL terms')


NLTerm(uAA,uAbar,TAA,TAbar,2*mc,-mc)
gamma_AA_u = np.vdot(uA_adj['c'],-NL_u_field['c'])
gamma_AA_T = np.vdot(TA_adj['c'],-NL_T_field['c'])
gamma_AA = gamma_AA_u + gamma_AA_T
logger.info('gamma_AA={0:g}'.format(gamma_AA))

gamma_u_AA = (-NL_u_field).evaluate().copy()
gamma_T_AA = (-NL_T_field).evaluate().copy()

NLTerm(uAAbar,uAm,TAAbar,TAm,0,1.*mc)
gamma_AAbar_u = np.vdot(uA_adj['c'],-NL_u_field['c'])
gamma_AAbar_T = np.vdot(TA_adj['c'],-NL_T_field['c'])
gamma_AAbar = gamma_AAbar_u + gamma_AAbar_T 
logger.info('gamma_AAbar={0:g}'.format(gamma_AAbar))

gamma_u_AAbar = (-NL_u_field).evaluate().copy()
gamma_T_AAbar = (-NL_T_field).evaluate().copy()

gamma = gamma_AA + gamma_AAbar
logger.info('gamma={0:g}'.format(gamma))

chi_u = u.copy()
chi_u_m = um.copy()

chi_u_m['g'] = (Rayleigh*Ekman*r_vec*TAm).evaluate()['g']
chi_u['g'] = chi_u_m['g']*np.exp(1j*mc*phi)
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

output_handler.add_task(gamma_u_AA,name='gamma_u_AA')
output_handler.add_task(gamma_T_AA,name='gamma_T_AA')
output_handler.add_task(gamma_u_AAbar,name='gamma_u_AAbar')
output_handler.add_task(gamma_T_AAbar,name='gamma_T_AAbar')

output_handler.add_task(chi_u_m,name='chi_u')
output_evaluator.evaluate_handlers(output_evaluator.handlers, timestep=0, wall_time=0, sim_time=0, iteration=0)
#, index=[1e-3],columns=['gamma_AA','gamma_AA_u','gamma_AA_T','gamma_AAbar','gamma_AAbar_u','gamma_AAbar_T','chi']
df = pd.DataFrame(np.array([gamma,gamma_AA,gamma_AA_u,gamma_AA_T,gamma_AAbar,gamma_AAbar_u,gamma_AAbar_T,chi]),index=['gamma','gamma_AA','gamma_AA_u','gamma_AA_T','gamma_AAbar','gamma_AAbar_u','gamma_AAbar_T','chi'],columns=[Ekman])
print(df)
df.to_csv('{0:s}/wnl_coefficients.csv'.format(file_dir))
