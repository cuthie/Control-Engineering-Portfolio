#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

class KalmanFilter {
  public:
    KalmanFilter(float dt, float q, float r) {
      this->dt = dt;

      // State estimate [position; velocity]
      x[0] = 0.0;
      x[1] = 0.0;

      // State covariance matrix
      P[0][0] = 1.0; P[0][1] = 0.0;
      P[1][0] = 0.0; P[1][1] = 1.0;

      // Process and measurement noise
      Q = q;
      R = r;
    }

    void predict(float u) {
      // State prediction
      float x0 = x[0] + dt * x[1];
      float x1 = x[1] + dt * u;

      // Covariance prediction
      float P00 = P[0][0] + dt * (P[1][0] + P[0][1]) + dt * dt * P[1][1] + Q;
      float P01 = P[0][1] + dt * P[1][1];
      float P10 = P[1][0] + dt * P[1][1];
      float P11 = P[1][1] + Q;

      x[0] = x0;
      x[1] = x1;

      P[0][0] = P00; P[0][1] = P01;
      P[1][0] = P10; P[1][1] = P11;
    }

    void update(float z) {
      // Measurement update
      float y = z - x[0]; // innovation
      float S = P[0][0] + R;
      float K0 = P[0][0] / S;
      float K1 = P[1][0] / S;

      x[0] += K0 * y;
      x[1] += K1 * y;

      // Covariance update
      float P00 = P[0][0] - K0 * P[0][0];
      float P01 = P[0][1] - K0 * P[0][1];
      float P10 = P[1][0] - K1 * P[0][0];
      float P11 = P[1][1] - K1 * P[0][1];

      P[0][0] = P00; P[0][1] = P01;
      P[1][0] = P10; P[1][1] = P11;
    }

    float getPosition() { return x[0]; }
    float getVelocity() { return x[1]; }

  private:
    float dt;
    float Q, R;
    float x[2];       // state estimate
    float P[2][2];    // covariance matrix
};

#endif