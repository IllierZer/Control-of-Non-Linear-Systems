import numpy as np
import matplotlib.pyplot as plt

class manipulator:
    def __init__(self, p_1, p_3, q_0, q_r):
        
        # system states
        self.q = q_0    
        self.q_r = q_r
        self.e = self.q - self.q_r
        self.q_dot = 0.
        self.q_dot_dot = 0.
        self.xi = 0.
        self.xi_dot = 0

        # constant values
        self.p_1 = p_1
        self.p_3 = p_3
        self.dt = 0.1

        self.update_mc()
    
    def update_mc(self):
        self.m = self.p_1 + (2 * self.p_3 * np.cos(self.q))
        self.c = -self.p_3 * np.sin(self.q) * self.q_dot
        self.m_dot = -2 * self.p_3 * np.sin(self.q) * self.q_dot
        self.c_dot = -(self.p_3 * np.cos(self.q) * (self.q_dot**2)) - (self.p_3 * np.sin(self.q) * self.q_dot_dot)
    
    def advance_state(self, tau):
        dt = self.dt

        # dynamics
        self.q_dot_dot = (self.xi - (self.c * self.q_dot)) / self.m
        self.xi_dot = tau        

        # update states
        self.q_dot += self.q_dot_dot * dt
        self.q += self.q_dot * dt
        self.xi += self.xi_dot * dt
        self.update_mc()

    def compute_control_law(self): # taking the control lyapunov function to be V(q,q_dot) = e^2/2 + q_dot^2/2
        # k_0 = Cq_dot + mq_r - mq - mq_dot
        k_0_dot = (self.c_dot * self.q_dot + self.c * self.q_dot_dot) + self.m_dot * self.q_r - (self.m_dot*self.q + self.m*self.q_dot) - (self.m_dot*self.q_dot + self.m*self.q_dot_dot)
        self.tau = k_0_dot - (2*self.q_dot/self.m) - (self.xi/(self.m**2)) + (self.c*self.q_dot/(self.m**2)) + ((self.q_r - self.q)/self.m)
        


    def simulate(self, t_f) :
        self.t = np.linspace(0,t_f,int(t_f/self.dt))
        self.compute_control_law()
        self.tau_array = []
        self.q_array = []
        self.q_r_array = []
        for _ in range (len(self.t)):
            self.tau_array.append(self.tau)
            self.q_array.append(self.q)
            self.q_r_array.append(self.q_r)
            self.advance_state(self.tau)
            self.compute_control_law()
        
    def plot(self):

        plt.subplots(1)
        plt.plot(self.t, self.tau_array, 'red')
        plt.xlabel("t")
        plt.ylabel(r"$\tau$")
        plt.title(r"control torque $\tau$ vs time t")
        plt.legend()
    
        plt.subplots(1)
        plt.plot(self.t, self.q_array, 'blue')
        plt.plot(self.t, self.q_r_array, 'k--', label = r"$q_r$", linewidth = 2)
        plt.xlabel("t")
        plt.ylabel(r"$q$")
        plt.title("angle q vs time t")
        plt.legend()
    
        plt.show()

if __name__ == '__main__' :
    # Roll No. : 200100085
    X = 5.
    p_1 = 3.31 + (X/30)
    p_3 = 0.16 + (X/400)

    q_r = 0.3
    q_0 = 1.

    m = manipulator(p_1,p_3,q_0,q_r)
    m.simulate(120)
    m.plot()

    # uncomment the lines below to see the convergence from differnet initial conditions
    # q_1 = 3.5
    # q_2 = -2.9
    # q_3 = 7
    # m1 = manipulator(p_1,p_3,q_0,q_r)
    # m1.simulate(120)
    # m2 = manipulator(p_1,p_3,q_1,q_r)
    # m2.simulate(120)
    # m3 = manipulator(p_1,p_3,q_2,q_r)
    # m3.simulate(120)
    # m4 = manipulator(p_1,p_3,q_3,q_r)
    # m4.simulate(120)

    # plt.plot(m1.t, m1.q_array, 'blue', label = "q_0 = 1")
    # plt.plot(m2.t, m2.q_array, 'green',label = "q_0 = 3.5")
    # plt.plot(m3.t, m3.q_array, 'cyan',label = "q_0 = -2.9")
    # plt.plot(m4.t, m4.q_array, 'magenta',label = "q_0 = 7")
    # plt.plot(m1.t, m1.q_r_array, 'k--', label = r"$q_r$", linewidth = 2)   
    # plt.xlabel("t")
    # plt.ylabel(r"$q$")
    # plt.title("Angle q vs time t")
    # plt.legend()
    # plt.show() 