import numpy as np

class KalmanFilter3D:
    def __init__(self, initial_state=None, process_noise=1e-2, measurement_noise=1e-1):
        # state: [x, y, z, vx, vy, vz]
        if initial_state is None:
            self.state = np.zeros(6)
        else:
            self.state = initial_state
        self.P = np.eye(6)
        self.F = np.eye(6)
        self.H = np.zeros((3, 6))
        self.H[:3, :3] = np.eye(3)
        self.Q = process_noise * np.eye(6)
        self.R = measurement_noise * np.eye(3)

    def predict(self, dt):
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt
        
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement):
        y = measurement - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.state += K @ y
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P