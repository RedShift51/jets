import os
import numpy as np
import pandas as pd
from scipy.integrate import RK45

class Solver():

    def __init__(self, kap=0.01, alp=0.01, r0=0.1, th0=0.5, phi0=0.1, \
                                        Pr0=0.6, Pth0=0.8, Pphi0=0.):
        self.kap = 0.01
        self.alp = 0.01
        self.r0 = r0
        self.th0 = th0
        self.phi0 = phi0
        self.Pr0 = Pr0
        self.Pth0 = Pth0
        self.Pphi0 = Pphi0
        self.t_beg = 0.
        self.t_end = 100.
        self.solver = RK45(self.system_sph, t0=self.t_beg, \
                y0=np.array([self.r0, self.th0, self.phi0, self.Pr0, self.Pth0, self.Pphi0]), \
                t_bound=self.t_end)

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

    def steps(self):
        t = [self.t_beg]
        r = [self.r0]
        th = [self.th0]
        phi = [self.phi0]
        Pr = [self.Pr0]
        Pth = [self.Pth0]
        Pphi = [self.Pphi0]
        while (r[-1] < 1. and t[-1] < self.t_end):
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

        return t, [r, th, phi, Pr, Pth, Pphi]

sol = Solver()
sol.steps()
