import numpy as np
import matplotlib.pyplot as plt

class two_link_manipulator:
    def __init__(self, p_1, p_2, p_3, q_r, q_0, D, K_p, k_1 = 1): # change the value of k_1 here and observe the effects
        # system states
        self.q = q_0 
        self.q_dot = np.zeros(np.shape(self.q),dtype = 'float64' )
        self.q_dot_dot = np.zeros(np.shape(self.q), dtype = 'float64')
        self.q_r = q_r
        self.e = self.q - self.q_r
        
        # constant values
        self.p_1 = p_1
        self.p_2 = p_2
        self.p_3 = p_3
        self.m_1 = 2.5
        self.m_2 = 1.5
        self.l = 0.4
        self.g = 9.81
        self.dt = 0.1

        # tunable parameters
        self.D = D # R^{2X2}
        self.K_p =  K_p # R^{2X2}
        self.k_1 = k_1 # control law gain
        self.update_qMCG()
    
    def update_qMCG(self):
        self.q_1 = self.q[0]
        self.q_2 = self.q[1]
        self.q_1_dot = self.q_dot[0]
        self.q_2_dot = self.q_dot[1]
        self.M = np.array([[self.p_1 + (2 * self.p_3 * np.cos(self.q_2)), self.p_2 + (self.p_3 * np.cos(self.q_2))],[(self.p_3 * np.cos(self.q_2)), self.p_2]])
        self.C = self.p_3 * np.sin(self.q_2) * np.array([[-self.q_2_dot, self.q_1_dot + self.q_2_dot],[self.q_1_dot, 0]])
        self.G = self.g * np.array([(self.l * (self.m_1 + self.m_2) * np.cos(self.q_1)) + (self.l * self.m_2 * np.cos(self.q_1 + self.q_2)), self.l * self.m_2 * np.cos(self.q_2)])      


    def advance_state(self):
        dt = self.dt

        # update states
        self.update_qMCG()
        self.e = self.q - self.q_r
        self.q_dot_dot = np.matmul(np.linalg.inv(self.M),(-self.k_1 * np.tanh(self.q_dot) - np.matmul(self.C,self.q_dot) - np.matmul(self.D, self.q_dot) - np.matmul(self.K_p ,self.e)))
        self.q_dot += self.q_dot_dot * dt
        self.q += self.q_dot * dt
    
    def simulate(self, t_f):
        self.t = np.linspace(0,t_f,int(t_f/self.dt))
        self.u_array = []
        self.q_array = []
        self.q_r_array = []
        self.q_1_array = []
        self.q_1_r_array = []
        self.q_2_array = []
        self.q_2_r_array = []
        self.u_1_array = []
        self.u_2_array = []
        for _ in range(len(self.t)):
            u = self.G - np.matmul(self.K_p , self.e) - self.k_1 * np.tanh(self.q_dot)
            self.u_array.append(np.linalg.norm(u))
            self.q_array.append(np.linalg.norm(self.q))
            self.q_r_array.append(np.linalg.norm(self.q_r))
            self.q_1_array.append(self.q[0])
            self.q_2_array.append(self.q[1])
            self.u_1_array.append(u[0])
            self.u_2_array.append(u[1])
            self.q_1_r_array.append(self.q_r[0])
            self.q_2_r_array.append(self.q_r[1])

            self.advance_state()
        

    def plot(self):
        
        plt.subplots(1)
        plt.plot(self.t, self.u_array, 'red')
        plt.xlabel("t")
        plt.ylabel("u")
        plt.title("Magnitude of u vs time t")
        plt.legend()
    
        plt.subplots(1)
        plt.plot(self.t, self.q_array, 'blue')
        plt.plot(self.t, self.q_r_array, 'k--', label = r"$q_r$", linewidth = 2)
        plt.xlabel("t")
        plt.ylabel(r"$q$")
        plt.title("Magnitude of q vs time t")
        plt.legend()
    
        plt.subplots(1)
        plt.plot(self.t, self.u_1_array, 'red', label = r"$u_1$")
        plt.plot(self.t, self.u_2_array, 'black', label = r"$u_2$")
        plt.xlabel("t")
        plt.title("Variation of Control Input components with time")
        plt.legend()
    
        plt.subplots(1)
        plt.plot(self.t, self.q_1_array, 'red', label = r"$q_1$")
        plt.plot(self.t, self.q_1_r_array, 'r--', label = r"$q_{1r}$")
        plt.plot(self.t, self.q_2_array, 'black', label = r"$q_2$")
        plt.plot(self.t, self.q_2_r_array, 'k--', label = r"$q_{2r}$")
        plt.xlabel("t")
        plt.title("Variation of State components with time")
        plt.legend()

        plt.show()
    
if __name__ == '__main__':
    # Roll No. : 200100085
    X = 5.
    p_1 = 3.31 + (X/30)
    p_2 = 0.116 + (X/500)
    p_3 = 0.16 + (X/400)

    q_r = np.array([0.4, 0], dtype = 'float64')
    q_0 = np.array([1, 2], dtype = 'float64')


    # D is a positive semi-definite matrix and K_p is a symmetric positive definite
    D = np.array([[1, 0 ],[0, 1]], dtype = 'float64')
    K_p = np.array([[1, 0 ],[0, 1]],dtype = 'float64')
    
    # tlm = two_link_manipulator(p_1, p_2, p_3, q_r, q_0, D, K_p)
    # tlm.simulate(25)
    # tlm.plot()

    # uncomment the below lines to see the convergence of different initial conditions
    q_1 = np.array([0,1], dtype = 'float64')
    q_2 = np.array([3,6], dtype = 'float64')
    q_3 = np.array([-1, 5], dtype = 'float64')
    tlm1 = two_link_manipulator(p_1, p_2, p_3, q_r, q_0, D, K_p)
    tlm1.simulate(25)
    tlm2 = two_link_manipulator(p_1, p_2, p_3, q_r, q_1, D, K_p)
    tlm2.simulate(25)
    tlm3 = two_link_manipulator(p_1, p_2, p_3, q_r, q_2, D, K_p)
    tlm3.simulate(25)
    tlm4 = two_link_manipulator(p_1, p_2, p_3, q_r, q_3, D, K_p)
    tlm4.simulate(25)

    plt.subplots(1)
    plt.plot(tlm1.t, tlm1.q_array, 'blue', label = "q_0 = [1, 2]")
    plt.plot(tlm2.t, tlm2.q_array, 'green',label = "q_0 = [0, 1]")
    plt.plot(tlm3.t, tlm3.q_array, 'cyan',label = "q_0 = [3, 6]")
    plt.plot(tlm4.t, tlm4.q_array, 'magenta',label = "q_0 = [-1, 5]")
    plt.plot(tlm1.t, tlm1.q_r_array, 'k--', label = r"$q_r$", linewidth = 2)
    plt.xlabel("t")
    plt.ylabel(r"$q$")
    plt.title("Magnitude of q vs time t")
    plt.legend()
    plt.show()






            



