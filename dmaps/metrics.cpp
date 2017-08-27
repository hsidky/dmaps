#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include "metrics.h"

using namespace Eigen;

namespace dmaps
{
	f_type rmsd(vector_t ri, vector_t rj, const vector_t& w)
	{
		// Maps for ease of use.
		Map<matrix3_t> xi(ri.data(), ri.size()/3, 3);
		Map<matrix3_t> xj(rj.data(), rj.size()/3, 3);

		// Subtract out centers of mass.
		f_type wtot = w.sum();
		vector3_t comi = (w.asDiagonal()*xi).colwise().sum()/wtot;
		vector3_t comj = (w.asDiagonal()*xj).colwise().sum()/wtot;
		xi.rowwise() -= comi.transpose(); 
		xj.rowwise() -= comj.transpose();

		// SVD of covariance matrix.
		matrix_t cov = xi.transpose()*w.asDiagonal()*xj;
		JacobiSVD<matrix_t> svd(cov, ComputeThinU | ComputeThinV);
		
		// Find rotation. 
		f_type d = (svd.matrixV()*svd.matrixU().transpose()).determinant() > 0 ? 1 : -1; 
		
		matrix33_t eye = matrix33_t::Identity(3, 3);
		eye(2, 2) = d;
		matrix33_t R = svd.matrixV()*eye*svd.matrixU().transpose();
		
		// Return rmsd.
		return std::sqrt((w.asDiagonal()*(xi - xj*R).array().square().matrix()).sum()/wtot);
	}
}