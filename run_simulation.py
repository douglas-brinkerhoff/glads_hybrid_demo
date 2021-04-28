import numpy as np
import matplotlib.pyplot as plt
import dolfin as df
import sys
import time
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--run_index',dest='run_index', type=int,default=99999)
parser.add_argument('--sheet_conductivity',dest='k_s', type=float, default=1e-3)
parser.add_argument('--channel_conductivity',dest='k_c', type=float, default=1e-1)
parser.add_argument('--bump_height',dest='h_r', type=float, default=1e-1)
parser.add_argument('--bump_slope',dest='delta_r', type=float,default=1e-1)
parser.add_argument('--bump_spacing',dest='l_r', type=float, default=1)
parser.add_argument('--basal_traction',dest='beta2', type=float, default=1e6)
parser.add_argument('--pressure_exponent',dest='p', type=float, default=1)
parser.add_argument('--sliding_exponent',dest='q', type=float, default=1)
parser.add_argument('--porosity',dest='e_v', type=float, default=1e-3)
parser.add_argument('--data_directory',dest='data_directory', default='data_low_resolution')
parser.add_argument('--results_directory',dest='results_directory', default='results_low_resolution')
parser.add_argument('--guess_directory',dest='guess_directory', default=None)

args = parser.parse_args()

time.sleep(np.random.rand()*3)

# Necessary for multipool execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

df.parameters['form_compiler']['optimize'] = True
df.parameters['form_compiler']['cpp_optimize'] = True
df.parameters['form_compiler']['quadrature_degree'] = 4
df.parameters['allow_extrapolation'] = True


# VERTICAL BASIS REPLACES A NORMAL FUNCTION, SUCH THAT VERTICAL DERIVATIVES
# CAN BE EVALUATED IN MUCH THE SAME WAY AS HORIZONTAL DERIVATIVES.  IT NEEDS
# TO BE SUPPLIED A LIST OF FUNCTIONS OF SIGMA THAT MULTIPLY EACH COEFFICIENT.
class VerticalBasis(object):
    def __init__(self,u,coef,dcoef):
        self.u = u
        self.coef = coef
        self.dcoef = dcoef

    def __call__(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.coef)])

    def ds(self,s):
        return sum([u*c(s) for u,c in zip(self.u,self.dcoef)])

    def dx(self,s,x):
        return sum([u.dx(x)*c(s) for u,c in zip(self.u,self.coef)])

# PERFORMS GAUSSIAN QUADRATURE FOR ARBITRARY FUNCTION OF SIGMA, QUAD POINTS, AND WEIGHTS
class VerticalIntegrator(object):
    def __init__(self,points,weights):
        self.points = points
        self.weights = weights
    def integral_term(self,f,s,w):
        return w*f(s)
    def intz(self,f):
        return sum([self.integral_term(f,s,w) for s,w in zip(self.points,self.weights)])

# This quadrature scheme is not high order (but still works).  Thanks to Mauro P. for point this out.
from numpy.polynomial.legendre import leggauss
def half_quad(order):
    points,weights = leggauss(order)
    points=points[(order-1)//2:]
    weights=weights[(order-1)//2:]
    weights[0] = weights[0]/2
    return points,weights

def Max(a, b): return (a+b+abs(a-b))/df.Constant(2)
def Min(a, b): return (a+b-abs(a-b))/df.Constant(2)
def Softplus(a,b,alpha=1): return Max(a,b) + 1./alpha*df.ln(1 + df.exp(-abs(a-b)*alpha))

# Here's where we specify where to look for simulation specific files in XML format.  
data_dir = args.data_directory
mesh = df.Mesh()
with df.XDMFFile(data_dir+"/mesh.xdmf") as infile:
    infile.read(mesh)
mesh.init()

class SpecFO(object):
    """ A class for solving the ansatz spectral in vertical-CG in horizontal Blatter-Pattyn equations """
    def __init__(self,mesh):
        # CG-1 vector element of size 4
        E_cg = df.FiniteElement("CG",mesh.ufl_cell(),1)
        self.elements = [E_cg]*4

    def set_coupler(self,coupler):
        # Inform the solver of the coupler class, which allows querying of effective pressure from
        # hydrology model.
        self.coupler = coupler

    def build_variables(self,n=3.0,g=9.81,rho_i=917.,rho_w=1000.,eps_reg=1e-5,Be=464158,p=1,q=1,beta2=2e-3,Nhat=1,Uhat=1,write_pvd=True,results_dir='./results/'):
        self.n = df.Constant(n)                     # Glen's exponent
        self.g = df.Constant(g)                     # gravity
        self.rho_i = df.Constant(rho_i)             # density
        self.rho_w = df.Constant(rho_w)
        self.eps_reg = df.Constant(eps_reg)
        self.Be = df.Constant(Be)                   # Ice viscosity coefficient
        self.p = df.Constant(p)                     # Pressure exponent
        self.q = df.Constant(q)                     # Sliding law exponent
        self.beta2 = df.Function(self.coupler.Q_r)  # Basal traction coefficient
        self.beta2.vector()[:] = beta2
        self.Nhat = Nhat                            # Pressure and velocity scales (see paper)
        self.Uhat = Uhat

        self.ubar = self.coupler.U[0]               # Individual degrees of freedom
        self.vbar = self.coupler.U[1]
        self.udef = self.coupler.U[2]
        self.vdef = self.coupler.U[3]

        self.u0_bar = df.Function(self.coupler.Q_cg) # Last time step's solution
        self.v0_bar = df.Function(self.coupler.Q_cg)
        self.u0_def = df.Function(self.coupler.Q_cg)
        self.v0_def = df.Function(self.coupler.Q_cg)

        self.previous_time_vars = [self.u0_bar,self.v0_bar,self.u0_def,self.v0_def]

        self.lamdabar_x = self.coupler.Lambda[0]   # Test functions
        self.lamdabar_y = self.coupler.Lambda[1]
        self.lamdadef_x = self.coupler.Lambda[2]
        self.lamdadef_y = self.coupler.Lambda[3]

        # TEST FUNCTION COEFFICIENTS
        coef = [lambda s:1.0, lambda s:1./(n+1)*((n+2)*s**(n+1) - 1)]  # These are the vertical basis functions
        dcoef = [lambda s:0, lambda s:(n+2)*s**n]                      # and symbolic derivatives
 
        u_ = [self.ubar,self.udef]
        v_ = [self.vbar,self.vdef]
        lamda_x_ = [self.lamdabar_x,self.lamdadef_x]
        lamda_y_ = [self.lamdabar_y,self.lamdadef_y]

        self.u = VerticalBasis(u_,coef,dcoef)
        self.v = VerticalBasis(v_,coef,dcoef)
        self.lamda_x = VerticalBasis(lamda_x_,coef,dcoef)
        self.lamda_y = VerticalBasis(lamda_y_,coef,dcoef)

        self.U_b = df.as_vector([self.u(1),self.v(1)]) 
 
        if write_pvd:
            self.write_pvd = True
            self.results_dir = results_dir
            self.Us = df.project(df.as_vector([self.u(0),self.v(0)]))
            self.Ub = df.project(df.as_vector([self.u(1),self.v(1)]))
            self.Us_file = df.File(results_dir+'U_s.pvd')
            self.Ub_file = df.File(results_dir+'U_b.pvd')

    def write_variables(self,t):
        # Helper function for writing current time step velocities to .pvd
        Us_temp = df.project(df.as_vector([self.u(0),self.v(0)]))
        Ub_temp = df.project(df.as_vector([self.u(1),self.v(1)]))

        self.Us.vector().set_local(Us_temp.vector().get_local())
        self.Ub.vector().set_local(Ub_temp.vector().get_local())
        self.Us_file << (self.Us,t)
        self.Ub_file << (self.Ub,t)

    def build_forms(self):
        # Assemble FEM forms
        n = self.n
        g = self.g
        rho_i = self.rho_i
        rho_w = self.rho_w 
        eps_reg = self.eps_reg
        Be = self.Be
        p = self.p
        q = self.q
        beta2 = self.beta2

        u = self.u
        v = self.v
        lamda_x = self.lamda_x
        lamda_y = self.lamda_y
        H = self.coupler.H_c
        B = self.coupler.B_c
        S = self.coupler.S
        N = self.coupler.hydro.N

        def dsdx(s):
            return 1./H*(S.dx(0) - s*H.dx(0))

        def dsdy(s):
            return 1./H*(S.dx(1) - s*H.dx(1))

        def dsdz(s):
            return -1./H 

        # 2nd INVARIANT STRAIN RATE
        def epsilon_dot(s):
            return ((u.dx(s,0) + u.ds(s)*dsdx(s))**2 \
                        +(v.dx(s,1) + v.ds(s)*dsdy(s))**2 \
                        +(u.dx(s,0) + u.ds(s)*dsdx(s))*(v.dx(s,1) + v.ds(s)*dsdy(s)) \
                        +0.25*((u.ds(s)*dsdz(s))**2 + (v.ds(s)*dsdz(s))**2 \
                        + ((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))**2) \
                        + eps_reg)

        # VISCOSITY
        def eta_v(s):
            return Be/2.*epsilon_dot(s)**((1.-n)/(2*n))

        # MEMBRANE STRESSES
        def membrane_xx(s):
            return (lamda_x.dx(s,0) + lamda_x.ds(s)*dsdx(s))*H*(eta_v(s))*(4*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 2*(v.dx(s,1) + v.ds(s)*dsdy(s)))

        def membrane_xy(s):
            return (lamda_x.dx(s,1) + lamda_x.ds(s)*dsdy(s))*H*(eta_v(s))*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))

        def membrane_yx(s):
            return (lamda_y.dx(s,0) + lamda_y.ds(s)*dsdx(s))*H*(eta_v(s))*((u.dx(s,1) + u.ds(s)*dsdy(s)) + (v.dx(s,0) + v.ds(s)*dsdx(s)))

        def membrane_yy(s):
            return (lamda_y.dx(s,1) + lamda_y.ds(s)*dsdy(s))*H*(eta_v(s))*(2*(u.dx(s,0) + u.ds(s)*dsdx(s)) + 4*(v.dx(s,1) + v.ds(s)*dsdy(s)))

        # SHEAR STRESSES
        def shear_xz(s):
            return dsdz(s)**2*lamda_x.ds(s)*H*eta_v(s)*u.ds(s)

        def shear_yz(s):
            return dsdz(s)**2*lamda_y.ds(s)*H*eta_v(s)*v.ds(s)

        # DRIVING STRESSES
        def tau_dx(s):
            return rho_i*g*H*S.dx(0)*lamda_x(s)

        def tau_dy(s):
            return rho_i*g*H*S.dx(1)*lamda_y(s)

        # GET QUADRATURE POINTS (THIS SHOULD BE ODD: WILL GENERATE THE GAUSS-LEGENDRE RULE 
        # POINTS AND WEIGHTS OF O(n), BUT ONLY THE POINTS IN [0,1] ARE KEPT< DUE TO SYMMETRY.
        points,weights = half_quad(9)

        # INSTANTIATE VERTICAL INTEGRATOR
        vi = VerticalIntegrator(points,weights)

        # Basal shear stress (note minimum effective pressure ~1m head)
        tau_bx = -beta2*(Max(N,10000)/self.Nhat)**p*abs((u(1)**2 + v(1)**2)/self.Uhat**2 + 1e-2)**((q-1)/2.)*u(1)/self.Uhat
        tau_by = -beta2*(Max(N,10000)/self.Nhat)**p*abs((u(1)**2 + v(1)**2)/self.Uhat**2 + 1e-2)**((q-1)/2.)*v(1)/self.Uhat

        # weak form residuals for BP approximation.
        R_u_body = (- vi.intz(membrane_xx) - vi.intz(membrane_xy) - vi.intz(shear_xz) + tau_bx*lamda_x(1) - vi.intz(tau_dx))*df.dx
        R_v_body = (- vi.intz(membrane_yx) - vi.intz(membrane_yy) - vi.intz(shear_yz) + tau_by*lamda_y(1) - vi.intz(tau_dy))*df.dx

        # Add residuals to coupler "master" residual
        self.coupler.R += R_u_body
        self.coupler.R += R_v_body

class GLADS(object):
    """ A class implementing GLADS from Werder 2012 """

    def __init__(self,mesh):
        E_cg = df.FiniteElement("CG",mesh.ufl_cell(),1) # CG element for pressure
        E_cr = df.FiniteElement("CR",mesh.ufl_cell(),1) # CR element (edgewise) for channel size
        E_dg = df.FiniteElement("DG",mesh.ufl_cell(),0) # DG element for cavity size
        self.output_holder = df.MeshFunction('double',mesh,1)

        # Mixed element combining above elements
        self.elements = [E_cg,E_dg,E_cr]

    def set_coupler(self,coupler):
        # Inform class of coupler class, such that it can access basal velocities, etc.
        self.coupler = coupler

    def build_variables(self,n=3.0,g=9.81,rho_i=917.,rho_w=1000.,eps_reg=1e-5,Be=214000,p=1,q=1,beta2=2e-3,La=3.35e5,G=0.042*60**2*24*365,k_s=1e-3*60**2*24*365,k_c=1e-1*60**2*24*365,e_v=1e-3,ct=7.5e-8,cw = 4.22e3,alpha=5./4.,beta=3./2.,k=3e5,h_r=0.02,delta_r=0.1,l_r=0.3,A=2e-17,theta=1.0,dt=1e-3,write_pvd=True,results_dir='./results/'):
        self.n = df.Constant(n)                      # Glen's exponent
        self.g = df.Constant(g)                      # gravity
        self.rho_i = df.Constant(rho_i)              # densities
        self.rho_w = df.Constant(rho_w)
        self.eps_reg = df.Constant(eps_reg)
        self.La = df.Constant(La)                    # Latent heat
        self.Be = df.Constant(Be)                    # Viscosity factor
        self.p = df.Constant(p)                      # Pressure exponent
        self.q = df.Constant(q)                      # Velocity exponent
        self.beta2 = df.Function(self.coupler.Q_r)   # Basal traction
        self.beta2.vector()[:] = beta2
        self.G = df.Constant(G)                      # Geothermal heat flux
        self.e_v = df.Constant(e_v)                  # englacial porosity
        self.ct = df.Constant(ct)                    # 
        self.cw = df.Constant(cw)
        self.alpha = df.Constant(alpha)              # Darcy-Weisbach exponents
        self.beta = df.Constant(beta)
        self.k_c = df.Function(self.coupler.Q_r)     # channel conductivity
        self.k_c.vector()[:] = k_c
        self.k_s = df.Function(self.coupler.Q_r)     # sheet conductivity
        self.k_s.vector()[:] = k_s
        self.h_r = df.Function(self.coupler.Q_r)     # bump height
        self.h_r.vector()[:] = h_r
        self.l_r = df.Function(self.coupler.Q_r)     # bump wavelength (not used)
        self.l_r.vector()[:] = l_r
        self.delta_r = df.Function(self.coupler.Q_r) # bump aspect ratio
        self.delta_r.vector()[:] = delta_r
        self.A = df.Constant(A)                      # Ice hardness for hydro model
        self.theta = df.Constant(theta)              # implicitness in time stepping

        H = self.coupler.H_c                         # Geometry
        B = self.coupler.B_c

        self.phi = self.coupler.U[4]
        self.h = self.coupler.U[5]
        self.S = self.coupler.U[6]

        self.phi0 = df.Function(self.coupler.Q_cg)
        self.h0 = df.Function(self.coupler.Q_dg)
        self.S0 = df.Function(self.coupler.Q_cr)

        self.previous_time_vars = [self.phi0,self.h0,self.S0]

        self.xsi = self.coupler.Lambda[4]           # test functions
        self.psi = self.coupler.Lambda[5]
        self.w = self.coupler.Lambda[6]
 
        self.P_0 = rho_i*g*H                        # overburden
        self.phi_m = rho_w*g*B                      # bed potential
        self.P_w = self.phi - self.phi_m            # water pressure
        self.N = self.P_0 - self.P_w                # effective pressure

        self.dt = df.Constant(dt)

        if write_pvd:
            self.write_pvd = True
            self.results_dir = results_dir
            self.Nout = df.project(self.N,coupler.Q_cg)
            self.phi_file = df.File(results_dir+'phi.pvd')
            self.h_file = df.File(results_dir+'h.pvd')
            self.S_file = df.File(results_dir+'S.pvd')
            self.N_file = df.File(results_dir+'N.pvd')
           
    def write_variables(self,t):
        N_temp = df.project(self.N,coupler.Q_cg)
        self.Nout.vector().set_local(N_temp.vector().get_local())
        self.h_file << (self.h0,t)
        self.phi_file << (self.phi0,t)
        self.N_file << (self.Nout,t)  
        for f in df.facets(mesh):
            p = f.midpoint()
            self.output_holder[f] = self.S0(p.x(),p.y()) 
        self.S_file << (self.output_holder,t) 


    def set_boundary_labels(self,edgefunction):
        self.edgefunction = edgefunction
      
    def init_variables(self,phi_init=1e-4,S_init=1.e-3,h_init=1e-2):
        # Set initial condition values
        self.h0.vector()[:] = h_init
        self.phi0.vector()[:] = self.rho_w*self.g*self.coupler.B_c.vector()[:] + 1.0#0.8*self.rho_i/self.rho_w*self.coupler.H_c.vector()[:]
        self.S0.vector()[:] = S_init

    def set_timestep(self,dt):
        # update timestep for variable time stepping
        if self.dt:
            self.dt.assign(dt)
        else:
            self.dt = df.Constant(dt)

    def init_dirichlet_bcs(self):
        # Set dirichlet boundary condition on pressure at the margin (not used)
        self.bcs = [df.DirichletBC(coupler.V.sub(4),df.project(self.rho_w*self.g*self.coupler.B_c,self.coupler.Q_cg),self.edgefunction,1)]

    def build_forms(self):
        n = self.n
        g = self.g
        rho_i = self.rho_i
        rho_w = self.rho_w 
        eps_reg = self.eps_reg
        Be = self.Be
        La = self.La
        p = self.p
        q = self.q
        beta2 = self.beta2
        G = self.G
        e_v = self.e_v
        ct = self.ct
        cw = self.cw
        alpha = self.alpha
        beta = self.beta
        k_c = self.k_c
        k_s = self.k_s
        h_r = self.h_r 
        delta_r = self.delta_r
        l_r = self.l_r
        A = self.A
        theta = self.theta

        normal = df.FacetNormal(mesh)
        edgefunction = self.edgefunction
        ds = df.ds(subdomain_data=edgefunction)

        h = self.h
        phi = self.phi
        S = self.S

        h0 = self.h0
        phi0 = self.phi0
        S0 = self.S0

        xsi = self.xsi
        psi = self.psi
        w = self.w

        dt = self.dt

        m = self.coupler.m

        N = self.N
        P_w = self.P_w

        # Edge-tangent unit vector
        s = df.as_vector([normal[1],-normal[0]])

        # derivative of hydraulic potential along edges
        dphids = df.dot(s,df.grad(phi))

        # derivative of test function along edges
        dxsids = df.dot(s,df.grad(xsi))

        # derivative water pressure along edges
        dPds = df.dot(s,df.grad(P_w))

        # Edgewise flux
        Q = -k_c*Max(S,1e-2)**alpha*(dphids**2 + 1.)**(beta/2. - 1)*dphids
        self.Q = Q

        # Edgewise background flux 
        q_c = -k_s*Max(h,1e-3)**alpha*(dphids**2 + 1.)**(beta/2. - 1)*dphids
        q = -k_s*Max(h,1e-3)**alpha*(df.dot(df.grad(phi),df.grad(phi)) + 1.)**(beta/2.-1)*df.grad(phi)

        l_c = 2.0

        # Channel melt rates
        Chi = abs(Q*dphids) + abs(l_c*q_c*dphids)
        Pi = -ct*cw*rho_w*Q*dPds

        k_e = df.Constant(1.0)

        # log-normal bump size
        sigma_hr = 1
        log_h_r_s = np.array([hh for hh in np.log(self.h_r(0,0)) + sigma_hr*np.linspace(-3,3,15)])
        probs = np.exp(-0.5*(log_h_r_s - np.log(self.h_r(0,0)))**2/(sigma_hr)**2)
        probs/=probs.sum()
        h_r_s = [df.Constant(np.exp(lhr)) for lhr in log_h_r_s]

        u_b = df.sqrt(self.coupler.stokes.u(1)**2 + self.coupler.stokes.v(1)**2 + 1e-2)

        # Opening rate (cavities)
        O = sum([p*delta_r*u_b*Max(1 - h/h_r_i,0) for p,h_r_i in zip(probs,h_r_s)])

        # Closing rate (cavities)
        C = A*h*abs(N)**(n-1)*Max(N,1000)

        # Closing rate (channels)
        C_c = A*S*abs(N)**(n-1)*Max(N,1000)

        # hydropotential residual
        R_phi = (xsi*e_v/(rho_w*g)*(phi-phi0)/dt - df.dot(df.grad(xsi),q) - xsi*(m + C - O - k_e*Max(phi-self.phi_m-rho_w/rho_i*self.P_0,0)))*df.dx + df.avg(-dxsids*Q + xsi*(Chi - Pi)/La*(1./rho_i - 1/rho_w) - xsi*C_c)*df.dS(metadata={'quadrature_degree':1})
        
        # Cavity residual
        R_h = ((h - h0)/dt - O + C)*psi*df.dx 

        # Channel residual (note reduced quadrature degree, which only performs integration at edge midpoints, to make CR edge hack work)
        R_S = df.avg(((S - S0)/dt - (Chi-Pi)/(La*rho_i) + C_c)*w)*df.dS(metadata={'quadrature_degree':1}) + S*w*ds

        # Add residuals to coupler "master" residual
        self.coupler.R += R_phi + R_h + R_S

        # Helper function for assessing edge flux.
        dQ = df.TrialFunction(self.coupler.Q_cr)
        p = df.TestFunction(self.coupler.Q_cr)
        self.R_Q = df.avg((dQ-abs(Q))*p)*df.dS + (dQ-abs(Q))*p*df.ds  

class Coupler(object):
    """ A class for holding geometry and allowing velocity and hydrology solvers to communicate """
    def __init__(self,mesh,stokes,hydro):

        self.stokes = stokes
        self.hydro = hydro

        elements = self.stokes.elements + self.hydro.elements
        E_V = df.MixedElement(elements)  
        self.V = df.FunctionSpace(mesh,E_V)

        E_cg = df.FiniteElement("CG",mesh.ufl_cell(),1)
        self.Q_cg = df.FunctionSpace(mesh,E_cg)
        
        E_cr = df.FiniteElement("CR",mesh.ufl_cell(),1)
        self.Q_cr = df.FunctionSpace(mesh,E_cr)
     
        E_dg = df.FiniteElement("DG",mesh.ufl_cell(),0)
        self.Q_dg = df.FunctionSpace(mesh,E_dg)

        E_r = df.FiniteElement('R',mesh.ufl_cell(),0)
        self.Q_r = df.FunctionSpace(mesh,E_r)
        self.dw = df.TestFunction(self.Q_r)

        self.space_list = [df.FunctionSpace(E,mesh) for E in elements]

        self.U = df.Function(self.V)
        self.Lambda = df.Function(self.V)
        self.Phi = df.TestFunction(self.V)
        self.dU = df.TrialFunction(self.V)

        self.R = 0

    def set_geometry(self,B_c,B_d,H_c,H_d):
        self.B_c = B_c
        self.H_c = H_c
        self.B_d = B_d
        self.H_d = H_d
        self.S = B_c + H_c

    def set_forcing(self,m):
        self.m = m

    def generate_forward_model(self):
        
        function_spaces = [u.function_space() for u in self.stokes.previous_time_vars+self.hydro.previous_time_vars]
        self.assigner_inv = df.FunctionAssigner(function_spaces,self.V)
        self.assigner     = df.FunctionAssigner(self.V,function_spaces)

        self.R_fwd = df.derivative(self.R,self.Lambda,self.Phi)
        self.J_fwd = df.derivative(self.R_fwd,self.U,self.dU)

    def generate_adjoint_model(self,U_obs): # Not used, but single time step adjoint exists here.
        self.R += ((U_obs[0] - self.stokes.u(0))**2)*df.dx + ((U_obs[1] - self.stokes.v(0))**2)*df.dx
        self.R_adj = df.derivative(self.R,self.U,self.Phi)
        self.J_adj = df.derivative(self.R_adj,self.Lambda,self.dU)
 
# Instantiate stokes solver
stokes = SpecFO(mesh)

# Instantiate hydrology solver
hydro = GLADS(mesh)

# Instantiate coupler
coupler = Coupler(mesh,stokes,hydro)

# Read geometry and observed velocities
B_c = df.Function(coupler.Q_cg,data_dir+'/Bed/B_c.xml')
H_c = df.Function(coupler.Q_cg,data_dir+'/Thk/H_c.xml')
B_d = df.Function(coupler.Q_dg,data_dir+'/Bed/B_d.xml')
H_d = df.Function(coupler.Q_dg,data_dir+'/Thk/H_d.xml')

u_obs = df.Function(coupler.Q_cg,data_dir+'/Vel_x/u_c.xml')
v_obs = df.Function(coupler.Q_cg,data_dir+'/Vel_y/v_c.xml')
U_obs = df.as_vector([u_obs,v_obs])

# Enforce a minimum thickness of 10m for numerical convenience
thklim = 10
Htemp = H_c.vector().get_local()
Htemp[Htemp<thklim] = thklim
H_c.vector().set_local(Htemp)
Htemp = H_d.vector().get_local()
Htemp[Htemp<thklim] = thklim
H_d.vector().set_local(Htemp)

# Read in SMB values
m = df.Function(coupler.Q_cg,data_dir+'/SMB/smb_mean_c.xml')
m.vector()[:] += 1e-2
m_vals = [df.Function(coupler.Q_cg,data_dir+'/SMB/smb_{}_c.xml'.format(i)) for i in range(12)]

# Set geometry and inform physics modules of coupling
coupler.set_geometry(B_c,B_d,H_c,H_d)
coupler.set_forcing(m)
stokes.set_coupler(coupler)
hydro.set_coupler(coupler)

# Convenience functions
ones = df.Function(coupler.Q_cg)
ones.vector()[:] = 1.
area = df.assemble(ones*df.dx)

# hydrology FacetFunction, where value is 2 if free-flux, 1 if atmospheric, zero otherwise.
edgefunction = df.MeshFunction('size_t',mesh,1)
for f in df.facets(mesh):
    if f.exterior():
        edgefunction[f] = 2
        if H_c(f.midpoint().x(),f.midpoint().y())<50:
            edgefunction[f]=1
        if m(f.midpoint().x(),f.midpoint().y())<0.2:
            edgefunction[f]=3

hydro.set_boundary_labels(edgefunction)

# Set initial condition on phi as 9/10 of overburden plus bedrock potential
phi_init = df.project(917*9.81*B_c + 0.9*1000*9.81*H_c)

H_mean = df.assemble(H_c*df.dx)/area
U_mean = 50.0

# Initialize physics classes
stokes.build_variables(results_dir=args.results_directory+'/run_{}/'.format(args.run_index),beta2=args.beta2,p=args.p,q=args.q,Nhat=df.Constant(917*9.81*H_mean),Uhat=U_mean)
hydro.build_variables(dt=0.0001,results_dir=args.results_directory+'/run_{}/'.format(args.run_index),k_s=args.k_s*60**2*24*365,k_c=args.k_c*60**2*24*365,h_r=args.h_r,delta_r=args.delta_r,l_r=args.l_r,beta2=args.beta2,p=args.p,q=args.q,e_v=args.e_v)
hydro.init_variables(S_init=1.0)

hydro.phi0.vector()[:] = phi_init.vector()[:]

stokes.build_forms()
hydro.build_forms()
coupler.generate_forward_model()
coupler.generate_adjoint_model(U_obs)

hydro.init_dirichlet_bcs()

# Setup timestepping
t = 0
t_end = 12.0

timestep_increase_fraction = 1.1  # Fraction the timestep increases upon each successful time step
timestep_reduction_fraction = 0.5 # Fraction the timestep decreases upon non-convergence

dt_max = 1./24.                   # Max timestep (half month)

converged = 1
t0 = time.time()                  # For run timing

# Hack to initialize a function to hold channel fluxes
Q_file = df.File(args.results_directory+'/run_{}/Q.pvd'.format(args.run_index))
Q = df.Function(coupler.Q_cr)
Q_function = df.MeshFunction('double',mesh,1)
phi_Q = df.TestFunction(coupler.Q_cr)
dQ = df.TrialFunction(coupler.Q_cr)
R = df.avg(phi_Q*dQ - phi_Q*abs(hydro.Q))*df.dS(metadata={'quadrature_degree':1}) + (phi_Q*dQ)*df.ds(metadata={'quadrature_degree':1})

rmse_phi = np.inf
steady_achieved=1

# Loop over time
while t<t_end:

    # Kill the sim if things are hopeless
    if hydro.dt(0)<1e-8:
        converged = 0
        break

    # Kill the sim if things take too long
    if (time.time()-t0)>72000:
        converged = 0 
        break

    try:
        # This can be used to determine whether or not to solve for a steady state or use time-varying meltwater flux
        if steady_achieved:
            month = (t%1)*12
            month_floor = np.floor(month)
            month_ceil = np.ceil(month)
            floor_weight = month_ceil - month 
            ceil_weight = month - month_floor

            coupler.m.vector()[:] = m_vals[np.int(month_floor%12)].vector()[:]*floor_weight + m_vals[np.int(month_ceil%12)].vector()[:]*ceil_weight

        # Instantiate variational problem and solver
        problem = df.NonlinearVariationalProblem(coupler.R_fwd,coupler.U,J=coupler.J_fwd)
        solver = df.NonlinearVariationalSolver(problem)
        solver.parameters['nonlinear_solver'] = 'newton'
        solver.parameters['newton_solver']['relaxation_parameter'] = 0.7
        solver.parameters['newton_solver']['relative_tolerance'] = 1e-3
        solver.parameters['newton_solver']['absolute_tolerance'] = 1e0
        solver.parameters['newton_solver']['error_on_nonconvergence'] = True
        solver.parameters['newton_solver']['linear_solver'] = 'mumps'
        solver.parameters['newton_solver']['maximum_iterations'] = 25
        solver.parameters['newton_solver']['report'] = False
        # Set previous time step values
        coupler.assigner.assign(coupler.U,stokes.previous_time_vars+hydro.previous_time_vars)
        # Solve equations
        solver.solve()

        # Write solution to previous time step variables
        coupler.assigner_inv.assign(stokes.previous_time_vars+hydro.previous_time_vars,coupler.U)

        # Write to pvd
        stokes.write_variables(t)
        hydro.write_variables(t)

        # Solve for channel flux and write
        df.solve(df.lhs(R)==df.rhs(R),Q)
        for f in df.facets(mesh):
            Q_function[f] = Q(f.midpoint().x(),f.midpoint().y())
        Q_file << (Q_function,t)

        # Increment time step
        t += hydro.dt(0)

        # Print current index, current time step size, current time
        print(args.run_index,hydro.dt(0),t)

        # Update time step
        hydro.dt.assign(min(hydro.dt(0)*timestep_increase_fraction,dt_max))
    except RuntimeError:
        # If solver fails, try again with a smaller time step
        hydro.dt.assign(hydro.dt(0)*timestep_reduction_fraction)
        print('Convergence not achieved.  Reducing time step to {0} and trying again'.format(hydro.dt(0)))

# Write some summary statistics to file
openfile = open(args.results_directory+'/run_{}/misfit.txt'.format(args.run_index),'w')
openfile.write(str(args.run_index)+' '+str(rmse_u)+' '+str(converged)+' '+str(t)+' '+str(time.time()-t0)+' '+str(rmse_phi))
openfile.close()
stokes.write_variables(t)
hydro.write_variables(t)
df.solve(df.lhs(R)==df.rhs(R),Q)
for f in df.facets(mesh):
    Q_function[f] = Q(f.midpoint().x(),f.midpoint().y())
Q_file << (Q_function,t)

# Save final time step solutions
df.File(args.results_directory+'/run_{}/U_final.xml'.format(args.run_index)) << coupler.U

    

        
