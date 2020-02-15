import os
import numpy as np
import pandas as pd
from scipy.integrate import RK45
import matplotlib.pyplot as plt

class Solver():

    def __init__(self, kap=0.01, alp=0.01, beta=0.01, r0=1., th0=0.5, phi0=0.1, \
                                        Pr0=0.8, Pth0=0., Pphi0=0.6):
        self.kap = kap
        self.alp = alp
        self.beta = beta

        self.r0 = r0
        self.th0 = th0
        self.phi0 = phi0
        self.Pr0 = Pr0
        self.Pth0 = Pth0
        self.Pphi0 = Pphi0
        
        
        self.t_beg = 0.
        self.t_end = 234.
        self.solver = RK45(self.system_sph, t0=self.t_beg, \
                y0=np.array([self.r0, self.th0, self.phi0, self.Pr0, self.Pth0, self.Pphi0]), \
                t_bound=self.t_end)

    """ Part related to sphere """
    def sum_gamma(self, *args):
        print(type(args[0]), args[0])
        ans_predv = [k*k for k in args]
        ans_vec = ans_predv[0]
        #print(type(ans_vec))
        try:
            for s in args[1:]:
                ans_vec += s
        except:
            return ans_vec
        return ans_vec

    def sum_sqrt(self, *args):
        return np.sqrt(np.sum([k*k for k in args]))

    def der_r_sph(self, r, th, phi, Pr, Pth, Pphi):
        return self.kap * Pr / self.sum_sqrt(Pr, Pth, Pphi)

    def der_th_sph(self, r, th, phi, Pr, Pth, Pphi):
        return self.kap * Pth / r / self.sum_sqrt(Pr, Pth, Pphi)

    def der_phi_sph(self, r, th, phi, Pr, Pth, Pphi):
        return self.kap * Pphi / r / np.sin(th) / self.sum_sqrt(Pr, Pth, Pphi)

    def der_Pr_sph(self, r, th, phi, Pr, Pth, Pphi):
        enum = self.kap * Pth**2 + self.kap * Pphi**2 + self.alp * Pth * np.sign(np.cos(th))
        return enum / self.sum_sqrt(Pr, Pth, Pphi) / r

    def der_Pth_sph(self, r, th, phi, Pr, Pth, Pphi):
        return (-self.kap*Pr*Pth + self.kap*(Pphi**2)/np.tan(th)) / r / self.sum_sqrt(Pr, Pth, Pphi)\
            - np.sin(th)*np.sign(np.cos(th))/r \
            + np.sign(np.cos(th))*(Pphi / r - self.alp * Pr) / r / self.sum_sqrt(Pr, Pth, Pphi)

    def der_Pphi_sph(self, r, th, phi, Pr, Pth, Pphi):
        return (-self.kap * (Pr*np.sin(th) + Pth*np.cos(th)) * Pphi / np.sin(th) - \
                Pth*np.sign(np.cos(th)) / r) / r / self.sum_sqrt(Pr, Pth, Pphi)

    def system_sph(self, t, y):
        r, th, phi, Pr, Pth, Pphi = y
        return np.array([self.der_r_sph(r, th, phi, Pr, Pth, Pphi), \
                        self.der_th_sph(r, th, phi, Pr, Pth, Pphi), \
                        self.der_phi_sph(r, th, phi, Pr, Pth, Pphi), \
                        self.der_Pr_sph(r, th, phi, Pr, Pth, Pphi), \
                        self.der_Pth_sph(r, th, phi, Pr, Pth, Pphi), \
                        self.der_Pphi_sph(r, th, phi, Pr, Pth, Pphi)])

    """ Part related to cylinder """
    def der_r_cyl(self, r, th, phi, Pr, Pth, Pphi):
        return Pr / self.sum_sqrt(Pr, Pth, Pphi)

    def der_z_cyl(self, r, z, phi, Pr, Pz, Pphi):
        return Pz / self.sum_sqrt(Pr, Pth, Pphi)

    def der_phi_cyl(self, r, z, phi, Pr, Pz, Pphi):
        return Pphi / self.sum_sqrt(Pr, Pz, Pphi)

    def der_Pr_cyl(self, r, z, phi, Pr, Pz, Pphi):
        cond = (1.-r**2) if (r < 1) else 0.
        return (self.kap * Pphi**2 / r + Pphi - Pz*self.alp*r*cond) / self.sum_sqrt(Pr, Pphi, Pz) +\
                r*cond*self.beta

    def der_Pphi_cyl(self, r, z, phi, Pr, Pz, Pphi):
        return (-Pr*Pphi/r - Pr) / self.sum_sqrt(Pr, Pphi, Pz)

    def der_Pz_cyl(self, r, z, phi, Pr, Pz, Pphi):
        cond = (1.-r**2) if (r<1) else 0.
        return self.alp*Pr*r*cond / self.sum_sqrt(Pr, Pphi, Pz)

    def system_cyl(self, t, y):
        r, z, phi, Pr, Pz, Pphi = y
        return np.array([self.der_r_cyl(r, z, phi, Pr, Pz, Pphi), \
                        self.der_z_cyl(r, z, phi, Pr, Pz, Pphi), \
                        self.der_phi_cyl(r, z, phi, Pr, Pz, Pphi), \
                        self.der_Pr_cyl(r, z, phi, Pr, Pz, Pphi), \
                        self.der_Pz_cyl(r, z, phi, Pr, Pz, Pphi), \
                        self.der_Pphi_cyl(r, z, phi, Pr, Pz, Pphi)])

    """ Rhunge steps """
    def steps(self):
        t = [self.t_beg]
        r = [self.r0]
        th = [self.th0]
        phi = [self.phi0]
        Pr = [self.Pr0]
        Pth = [self.Pth0]
        Pphi = [self.Pphi0]
        while (r[-1]*np.sin(th[-1]) <= 1. and t[-1] < self.t_end):
            self.solver.step()
            curr_sol = self.solver.dense_output()
            r_new, th_new, phi_new, Pr_new, Pth_new, Pphi_new = curr_sol(self.solver.t)
            
            t.append([curr_sol.t][:][0])
            r.append([r_new][:][0])
            th.append([th_new][:][0])
            phi.append([phi_new][:][0])
            Pr.append([Pr_new][:][0])
            Pth.append([Pth_new][:][0])
            Pphi.append([Pphi_new][:][0])
        
        if (r[-1]*np.sin(th[-1]) >= 1. and t[-1] < self.t_end):
            solver_cyl = RK45(self.system_cyl, t0=self.t_beg, \
                y0=np.array([r[-1], z[-1], phi[-1], Pr[-1], self.Pth0, self.Pphi0]), \
                t_bound=self.t_end*2)

        return np.array(t), [np.array(r), np.array(th), np.array(phi), \
                np.array(Pr), np.array(Pth), np.array(Pphi)]

sol = Solver()
ans = sol.steps()

print(ans[1][0], ans[1][3], ans[1][4], ans[1][5])
plt.plot(ans[1][0]*np.sin(ans[1][1]), sol.sum_gamma(ans[1][3], ans[1][4], ans[1][5]))
plt.show()
