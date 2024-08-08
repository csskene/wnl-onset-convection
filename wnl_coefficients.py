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
    --test                        Test 2.5D calculation with full 3D residual calculation (expensive!)
    --internal                    Whether to use internal heating or not
"""

import numpy as np
import dedalus.public as d3
import time
import h5py
import logging
import dedalus.core.evaluator as evaluator
from docopt import docopt
import pandas as pd

logger = logging.getLogger(__name__)

args = docopt(__doc__)
# Parameters
Lmax             = int(args['--Lmax'])
Nmax             = int(args['--Nmax'])
Ekman            = float(args['--Ekman'])
Prandtl          = float(args['--Prandtl'])
test             = args['--test']

beta =  float(args['--beta'])

internal = args['--internal']

r_outer = 1/(1-beta)
r_inner = r_outer - 1

file_dir = 'data/Ekman_{0:g}_Prandtl_{1:g}_beta_{2:g}_internal_{3:s}'.format(Ekman, Prandtl, beta, str(internal))

data = np.load('{0:s}/results.npz'.format(file_dir))

idx = np.argmin(data['Ra'])
mc = data['ms'][idx]
Rayleigh_ = data['Ra'][idx]
om_A = np.squeeze(data['eigs'][idx]).imag
logger.info('Calculating coefficients for mc={0:d}, Ra={1:f}, om_A={2:f}'.format(mc, Rayleigh_, om_A))

radii = (r_inner,r_outer)

vol = 4*np.pi/3*(r_outer**3-r_inner**3)

ts = time.time()

c = d3.SphericalCoordinates('phi', 'theta', 'r')
d = d3.Distributor((c,), dtype=np.complex128)
b = d3.ShellBasis(c, (2*(2*mc+2),Lmax+1,Nmax+1), radii=radii, dealias=(1,1,1), dtype=np.complex128)
s2_basis = b.S2_basis()

b_inner = b.S2_basis(radius=r_inner)
b_outer = b.S2_basis(radius=r_outer)
phi, theta, r = d.local_grids(b)

u = d.VectorField(c, name='u', bases=b)
p = d.Field(name='p', bases=b)
T = d.Field(name='T', bases=b)
T0 = d.Field(name='T0', bases=b.meridional_basis)

tau_u1 = d.VectorField(c, name='tau_u1', bases=s2_basis)
tau_u2 = d.VectorField(c, name='tau_u2', bases=s2_basis)
tau_p = d.Field(name='tau_p')
tau_phi = d.Field(name='tau_phi')

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

lift_basis = b.clone_with(k=1) # First derivative basis
lift  = lambda A, n: d3.Lift(A, lift_basis, n)
integ = lambda A: d3.Integrate(A, c)
grad_u = d3.grad(u) + rvec*lift(tau_u1,-1) # First-order reduction
grad_T = d3.grad(T) + rvec*lift(tau_T1,-1) # First-order reduction

# Read in eigenvector
um = d.VectorField(c,name='um', bases=b.meridional_basis)
Tm = d.Field(name='Tm', bases=b.meridional_basis)

with h5py.File('{0:s}/eigenvector/eigenvector_s1.h5'.format(file_dir), mode='r') as input_file:
    for state_variable in [um, Tm]:
        state_variable.load_from_hdf5(input_file, -1)

u['g'] = um['g']*np.exp(1j*mc*phi)
T['g'] = Tm['g']*np.exp(1j*mc*phi)
uA = u.copy()
TA = T.copy()
uAm = um.copy()
TAm = Tm.copy()

urealm = uAm.copy()
urealm['g'] = uAm['g'].real

uimagm = uAm.copy()
uimagm['g'] = uAm['g'].imag

ureal = uA.copy()
ureal['g'] = uA['g'].real

norm = 0.5*(0.5*d3.integ(urealm@urealm + uimagm@uimagm)).evaluate()['g'][0,0,0]
logger.info('norm = %f' % norm.real)

# 2.5 operators
e_phi = d.VectorField(c,name='e_phi')
e_phi['g'][0]=1
e_theta = d.VectorField(c,name='e_theta')
e_theta['g'][1]=1
e_r = d.VectorField(c,name='e_r')
e_r['g'][2]=1

m_field = d.Field(name='m_field', bases=b.meridional_basis)
m_field['g'] = 1/(r*np.sin(theta))

curlm = lambda A, m: d3.curl(A) - e_r*(1j*m*m_field*e_theta@A) + e_theta*(1j*m*m_field*e_r@A)
lapm  = lambda A, m: -curlm(curlm(A,m),m)

e_r_phi = d.TensorField((c,c), name='e_r_phi')
e_r_phi['g'][2,0] = 1
e_theta_phi = d.TensorField((c,c), name='e_theta_phi')
e_theta_phi['g'][1,0] = 1
e_phi_phi = d.TensorField((c,c), name='e_phi_phi')
e_phi_phi['g'][0,0] = 1

gradm = lambda A,m: d3.grad(A) + m_field*1j*m*(e_r_phi*A@e_r + e_theta_phi*A@e_theta + e_phi_phi*A@e_phi )
if test:
    ## Check energy balance (3D calculation)
    D_nu = np.max(d3.integ(d3.dot(ureal,d3.lap(ureal))).evaluate()['g'])
    print(D_nu)
    strain = 0.5*(d3.grad(ureal) + d3.trans(d3.grad(ureal)))
    De = 2*np.max(d3.integ(d3.trace(strain@strain)).evaluate()['g'])
    print(De)
    residual = np.abs(D_nu+De)/np.max([np.abs(De),np.abs(D_nu)])
    logger.info('Residual_3D = {0:g}'.format(residual))

## Check energy balance
lap_term = lapm(um,mc).evaluate()
lap_real = urealm.copy()
lap_real['g'] = lap_term['g'].real

lap_imag = uimagm.copy()
lap_imag['g'] = lap_term['g'].imag
D_nu = np.max(0.5*d3.integ(urealm@lap_real + uimagm@lap_imag).evaluate()['g'])

strain = 0.5*(gradm(um,mc) + d3.trans(gradm(um,mc))).evaluate()
strain_real = d.TensorField((c,c), name='strain_real',bases=b)
strain_real['g'] = strain['g'].real
strain_imag = d.TensorField((c,c), name='strain_imag',bases=b)
strain_imag['g'] = strain['g'].imag
De = 2*np.max(0.5*d3.integ(d3.trace(strain_real@strain_real) + d3.trace(strain_imag@strain_imag)).evaluate()['g'])
residual = np.abs(D_nu+De)/np.max([np.abs(De),np.abs(D_nu)])
logger.info('Residual = {0:g}'.format(residual))

uAbar = uAm.copy()
uAbar['g'] = np.conj(uAm['g'])

TAbar = TAm.copy()
TAbar['g'] = np.conj(TAm['g'])

# Read in adjoint eigenvector
with h5py.File('{0:s}/eigenvector_adj/eigenvector_adj_s1.h5'.format(file_dir), mode='r') as input_file:
    for state_variable in [um,Tm]:
        state_variable.load_from_hdf5(input_file,-1)

u['g'] = um['g']*np.exp(1j*mc*phi)
T['g'] = Tm['g']*np.exp(1j*mc*phi)
uA_adj = u.copy()
TA_adj = T.copy()
uAm_adj = um.copy()
TAm_adj = Tm.copy()

term = 0
term += np.vdot(uA_adj['c'][:,mc,:,:],uA['c'][:,mc,:,:])
term += np.vdot(TA_adj['c'][mc,:,:],TA['c'][mc,:,:])

logger.info('Check adjoint normalisation = {0:f}'.format(term))

# Function to compute nonlinear terms
NL_u_field = d.VectorField(c, name='NL_u_field', bases=b)
NL_T_field = d.Field(name='NL_T_field', bases=b)
def NLTerm(u1, u2, T1, T2, m1, m2):
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
problem = d3.LBVP([p, u, T, tau_u1, tau_u2, tau_T1, tau_T2, tau_p], namespace=locals())
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

uAAbar = d.VectorField(c,name='uAAbar', bases=b.meridional_basis)
TAAbar = d.Field(name='TAAbar', bases=b.meridional_basis)
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

NLTerm(uAm, uAm, TAm, TAm, mc, mc)
problem = d3.LBVP([p, u, T, tau_u1, tau_u2, tau_T1, tau_T2, tau_p], namespace=locals())
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
uAA = d.VectorField(c,name='uAA', bases=b.meridional_basis)
TAA = d.Field(name='TAA', bases=b.meridional_basis)
uAA['g'][:,0,:,:] = u['g'][:,0,:,:]
TAA['g'][0,:,:] = T['g'][0,:,:]

##############################
## Compute WNL coefficients ##
##############################

logger.info('Computing WNL terms')

NLTerm(uAA, uAbar, TAA, TAbar, 2*mc, -mc)
gamma_AA_u = np.vdot(uA_adj['c'][:,mc,:,:],-NL_u_field['c'][:,mc,:,:])
gamma_AA_T = np.vdot(TA_adj['c'][mc,:,:],-NL_T_field['c'][mc,:,:])
gamma_AA = gamma_AA_u + gamma_AA_T
logger.info('gamma_AA={0:g}'.format(gamma_AA))

gamma_u_AA = um.copy()
gamma_T_AA = Tm.copy()
gamma_u_AA['g'][:,0,:,:] = (-NL_u_field).evaluate()['g'][:,0,:,:]
gamma_T_AA['g'][0,:,:]  = (-NL_T_field).evaluate()['g'][0,:,:]

NLTerm(uAAbar, uAm, TAAbar, TAm, 0, 1.*mc)
gamma_AAbar_u = np.vdot(uA_adj['c'][:,mc,:,:],-NL_u_field['c'][:,mc,:,:])
gamma_AAbar_T = np.vdot(TA_adj['c'][mc,:,:],-NL_T_field['c'][mc,:,:])
gamma_AAbar = gamma_AAbar_u + gamma_AAbar_T 
logger.info('gamma_AAbar={0:g}'.format(gamma_AAbar))

gamma_u_AAbar = um.copy()
gamma_T_AAbar = Tm.copy()
gamma_u_AAbar['g'][:,0,:,:] = (-NL_u_field).evaluate()['g'][:,0,:,:]
gamma_T_AAbar['g'][0,:,:]  = (-NL_T_field).evaluate()['g'][0,:,:]

gamma = gamma_AA + gamma_AAbar
logger.info('gamma={0:g}'.format(gamma))

chi_u = u.copy()
chi_u_m = um.copy()

chi_u_m['g'] = (Rayleigh*Ekman*r_vec*TAm).evaluate()['g']
chi_u['g'] = chi_u_m['g']*np.exp(1j*mc*phi)
chi = np.vdot(uA_adj['c'][:,mc,:,:],chi_u['c'][:,mc,:,:])
logger.info('chi={0:g}'.format(chi))

logger.info('Amplitude={0:g}'.format(np.sqrt(chi.real/gamma.real)))
logger.info('Total time taken={0:g}'.format(time.time()-ts))

################
## Save terms ##
################

output_evaluator = evaluator.Evaluator(d, locals())
output_handler = output_evaluator.add_file_handler('{0:s}/wnl_terms'.format(file_dir))

output_handler.add_task(uAA, name='uAA')
output_handler.add_task(TAA, name='TAA')

output_handler.add_task(uAAbar, name='uAAbar')
output_handler.add_task(TAAbar, name='TAAbar')

output_handler.add_task(gamma_u_AA, name='gamma_u_AA')
output_handler.add_task(gamma_T_AA, name='gamma_T_AA')
output_handler.add_task(gamma_u_AAbar, name='gamma_u_AAbar')
output_handler.add_task(gamma_T_AAbar, name='gamma_T_AAbar')

output_handler.add_task(chi_u_m, name='chi_u')
output_evaluator.evaluate_handlers(output_evaluator.handlers, timestep=0, wall_time=0, sim_time=0, iteration=0)
columns = ['gamma','gamma_AA','gamma_AA_u','gamma_AA_T','gamma_AAbar','gamma_AAbar_u','gamma_AAbar_T','chi','eig','residual','beta','Prandtl','mc','internal']
data = [gamma,gamma_AA,gamma_AA_u,gamma_AA_T,gamma_AAbar,gamma_AAbar_u,gamma_AAbar_T,chi,np.squeeze(data['eigs'][idx]),residual,beta,Prandtl,mc,internal]
frame_data = dict(zip(columns,data))

df = pd.DataFrame(data=frame_data,index=[Ekman])
print(df.T)
df.to_csv('{0:s}/wnl_coefficients.csv'.format(file_dir))
