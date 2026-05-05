/*
	FILE: kalman_filter.h
	--------------------------------------
	header of kalman_filter velocity estimator
*/

#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include <Eigen/Dense>

using Eigen::MatrixXd;

namespace onboardDetector{
    class kalman_filter
    {
        private:
        // members
        bool is_initialized;
        MatrixXd states;
        MatrixXd A; // state matrix
        MatrixXd B; // input matrix
        MatrixXd H; // observation matrix
        MatrixXd P; // uncertianty
        MatrixXd Q; // process noise
        MatrixXd R; // obsevation noise

        public:
        // constructor
        kalman_filter();

        // set up the filter
        void setup(const MatrixXd& states,
                   const MatrixXd& A,
                   const MatrixXd& B,
                   const MatrixXd& H,
                   const MatrixXd& P,
                   const MatrixXd& Q,
                   const MatrixXd& R);

        // set A (sometimes sampling time will differ)
        void setA(const MatrixXd& A);

        // state estimate
        void estimate(const MatrixXd& z, const MatrixXd& u);

        // predict only (no measurement update)
        void predict(const MatrixXd& u);

        // read output from the state
        double output(int state_index) const; // Maybe should not be const

        // set a single state component (used for velocity warm-start seeding)
        void setState(int state_index, double value);

        // Read a single covariance element P(i, j). Used by downstream
        // consumers (e.g. MPC) to size obstacle uncertainty inflation.
        // Returns 0.0 when uninitialized or out of bounds (caller treats
        // that as "no covariance info available").
        double covariance(int i, int j) const;
    };
}

#endif