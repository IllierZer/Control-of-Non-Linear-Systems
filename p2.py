import numpy as np
import matplotlib.pyplot as plt

class rigid_body:
    def __init__(self, J, rho_0, omega_0):
        self.J = J
        self.rho = rho_0
        self.rho_cross = np.array([[0, - self.rho[2], self.rho[1]],[self.rho[2], 0, -self.rho[0]],[-self.rho[1], self.rho[0], 0]])
        self.omega = omega_0
        self.I = np.identity(3)
        self.dt = 0.1
        self.k_1 = 1 # storage function gain
        self.k_2 = 12 # control law gain
        self.compute_dW_drho()
    
    # now compute the jacobian of the storage function to compute the control law
    def compute_dW_drho(self):
        self.dW_drho = (2 * self.k_1 / (1 + np.matmul(np.transpose(self.rho), self.rho))) * self.rho

    def advance_state(self, u) :
        dt = self.dt

        # dynamics
        self.rho_dot = np.matmul((self.I + self.rho_cross + np.matmul(self.rho, np.transpose(self.rho))), self.omega)
        self.omega_dot = np.matmul(np.linalg.inv(self.J),(-(np.cross(self.omega , np.matmul(self.J,self.omega))) + u))
        self.y = self.omega
        self.compute_dW_drho()
        
        # update states
        self.omega += self.omega_dot * dt
        self.rho += self.rho_dot * dt
        self.rho_cross = np.array([[0, - self.rho[2], self.rho[1]],[self.rho[2], 0, -self.rho[0]],[-self.rho[1], self.rho[0], 0]])
    
    def compute_control_law(self):
        self.nu = -self.k_2 * np.tanh(self.omega)
        self.u = self.nu - np.transpose(np.matmul(self.dW_drho,(self.I + self.rho_cross + np.matmul(self.rho, np.transpose(self.rho))))) 

    def simulate(self, t_f):
        self.t = np.linspace(0,t_f,int(t_f/self.dt))
        self.compute_control_law()
        self.u_array = []
        self.rho_array = []
        self.omega_array = []
        self.u_1_array = []
        self.u_2_array = []
        self.u_3_array = []
        self.rho_1_array = []
        self.rho_2_array = []
        self.rho_3_array = []
        self.omega_1_array = []
        self.omega_2_array = []
        self.omega_3_array = []

        for _ in range (len(self.t)):
            self.rho_array.append(np.linalg.norm(self.rho))
            self.omega_array.append(np.linalg.norm(self.omega))
            self.u_array.append(np.linalg.norm(self.u))
            self.rho_1_array.append(self.rho[0])
            self.rho_2_array.append(self.rho[1])
            self.rho_3_array.append(self.rho[2])
            self.omega_1_array.append(self.omega[0])
            self.omega_2_array.append(self.omega[1])
            self.omega_3_array.append(self.omega[2])
            self.u_1_array.append(self.u[0])
            self.u_2_array.append(self.u[1])
            self.u_3_array.append(self.u[2])
            self.advance_state(self.u)
            self.compute_control_law()

    def plot(self):
        # plot of control with respect to time
        plt.subplots(1)
        plt.plot(self.t, self.u_array, 'red')
        plt.xlabel("t")
        plt.ylabel("u")
        plt.title("Magnitude of u vs time t")
        plt.legend()

        # plot of omega with respect to time
        plt.subplots(1)
        plt.plot(self.t, self.omega_array, 'blue')
        plt.xlabel("t")
        plt.ylabel(r"$\omega$")
        plt.title(r"Magnitude of $\omega$ vs time t")
        plt.legend()
        
        # plot of omega with respect to time
        plt.subplots(1)
        plt.plot(self.t, self.rho_array, 'black')
        plt.xlabel("t")
        plt.ylabel(r"$\rho$")
        plt.title(r"Magnitude of $\rho$ vs time t")
        plt.legend()
    
        plt.subplots(1)
        plt.plot(self.t, self.rho_1_array, 'black', label = r"$\rho_1$")
        plt.plot(self.t, self.rho_2_array, 'red', label = r"$\rho_2$")
        plt.plot(self.t, self.rho_3_array, 'green', label = r"$\rho_3$")
        plt.xlabel("t")
        plt.ylabel(r"$\rho$")
        plt.title(r"Components of $\rho$ vs time t")
        plt.legend()

        plt.subplots(1)
        plt.plot(self.t, self.omega_1_array, 'black', label = r"$\omega_1$")
        plt.plot(self.t, self.omega_2_array, 'red', label = r"$\omega_2$")
        plt.plot(self.t, self.omega_3_array, 'green', label = r"$\omega_3$")
        plt.xlabel("t")
        plt.ylabel(r"$\omega$")
        plt.title(r"Components of $\omega$ vs time t")
        plt.legend()

        plt.subplots(1)
        plt.plot(self.t, self.u_1_array, 'black', label = r"$u_1$")
        plt.plot(self.t, self.u_2_array, 'red', label = r"$u_2$")
        plt.plot(self.t, self.u_3_array, 'green', label = r"$u_3$")
        plt.xlabel("t")
        plt.ylabel(r"$u$")
        plt.title(r"Components of $u$ vs time t")
        plt.legend()

        plt.show()

if __name__ == '__main__':
    # Roll_No := 200100085
    alpha_1 = 0.85
    J = np.array([[20 + alpha_1, 1.2, 0.9],[1.2, 17 + alpha_1, 1.4],[0.9, 1.4, 15 + alpha_1]])
    rho_0 = np.transpose(np.array([-0.02, 0, 0.045]))
    omega_0 = np.transpose(np.array([0.004, -0.007, 0.017]))
    rot_dyn = rigid_body(J, rho_0, omega_0)
    rot_dyn.simulate(35)
    rot_dyn.plot()