import numpy as np

class AdaptiveKalmanFilter:
    """
    Implements an Adaptive Kalman Filter (AKF) with Innovation-based tuning.
    Used to track 'Innovation' (regime shifts) and adjust Q (Process Noise).
    """
    def __init__(self, initial_state=100.0, initial_uncert=1.0, R=0.01, initial_Q=0.001):
        self.x = initial_state  # State estimate
        self.p = initial_uncert # Estimate uncertainty
        self.R = R              # Measurement noise covariance
        self.Q = initial_Q      # Process noise covariance
        self.innovation = 0.0
        self.s = 0.0            # Innovation covariance
        
    def update(self, measurement: float):
        """
        Standard Kalman update + Innovation-based Q-tuning.
        """
        # Time Update (Predict)
        # x_prev = x, p_prev = p + Q
        self.p = self.p + self.Q
        
        # Measurement Update (Correct)
        self.innovation = measurement - self.x
        self.s = self.p + self.R
        k = self.p / self.s  # Kalman gain
        
        self.x = self.x + k * self.innovation
        self.p = (1 - k) * self.p
        
        # Adaptive Q Tuning:
        # If innovation is high relative to expected variance (S), increase Q (be more curious/adaptive)
        # If innovation is low, decrease Q (be more stable)
        innovation_score = (self.innovation**2) / self.s
        
        # Exponential smoothing for Q adaptation
        alpha = 0.1
        target_Q = max(1e-6, min(0.5, 0.01 * innovation_score))
        self.Q = (1 - alpha) * self.Q + alpha * target_Q
        
        return self.x, self.Q, innovation_score

if __name__ == "__main__":
    akf = AdaptiveKalmanFilter()
    for m in [100.1, 100.2, 110.0, 115.0, 114.5]: # Simulate a shift at 110.0
        x, q, score = akf.update(m)
        print(f"Meas: {m:.1f}, Est: {x:.4f}, Q: {q:.6f}, Innovation Score: {score:.4f}")
