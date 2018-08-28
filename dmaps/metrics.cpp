#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include "metrics.h"
#include <iostream>

using namespace Eigen;

namespace dmaps
{
	f_type rmsd(const vector_t& ri, const vector_t& rj, const vector_t& w)
	{
		// Copy vectors for manipulation.
		vector_t r1 = ri, r2 = rj;

		// Get subset of weights just to make sure the size is proper. 
		const vector_t& ws = w.segment(0, ri.size()/3);

		// Maps for ease of use.
		Map<matrix3_t> xi(r1.data(), r1.size()/3, 3);
		Map<matrix3_t> xj(r2.data(), r2.size()/3, 3);

		// Subtract out centers of mass.
		f_type wtot = ws.sum();
		vector3_t comi = (ws.asDiagonal()*xi).colwise().sum()/wtot;
		vector3_t comj = (ws.asDiagonal()*xj).colwise().sum()/wtot;
		xi.rowwise() -= comi.transpose(); 
		xj.rowwise() -= comj.transpose();

		// SVD of covariance matrix.
		matrix_t cov = xi.transpose()*ws.asDiagonal()*xj;
		JacobiSVD<matrix_t> svd(cov, ComputeThinU | ComputeThinV);
		
		// Find rotation. 
		f_type d = (svd.matrixV()*svd.matrixU().transpose()).determinant() > 0 ? 1 : -1; 
		
		matrix33_t eye = matrix33_t::Identity(3, 3);
		eye(2, 2) = d;
		matrix33_t R = svd.matrixV()*eye*svd.matrixU().transpose();
		
		// Return rmsd.
		return std::sqrt((ws.asDiagonal()*(xi - xj*R).array().square().matrix()).sum()/wtot);
	}

	f_type euclidean(const vector_t& ri, const vector_t& rj, const vector_t& w)
	{
		return std::sqrt((w.array()*(ri-rj).array().square()).sum());
	}

	f_type contact_map(const vector_t& ri, const vector_t& rj, const vector_t&)
	{
		Map<const matrix3_t> xi(ri.data(), ri.size()/3, 3);
		Map<const matrix3_t> xj(rj.data(), rj.size()/3, 3);

		int irows = xi.rows(), jrows = xj.rows();
		vector_t dxi(irows*irows/2 - irows/2), dxj(jrows*jrows/2 - jrows/2);
		
		// Compute pairwise distances.
		int k = 0;
		for(int i = 0; i < irows - 1; ++i)
		{
			int nrows = irows - i - 1;
			dxi.segment(k, nrows) = (xi.bottomRows(nrows).rowwise() - xi.row(i)).matrix().rowwise().norm();
			k += nrows;
		}
		
		k = 0;
		for(int i = 0; i < jrows - 1; ++i)
		{
			int nrows = jrows - i - 1;
			dxj.segment(k, nrows) = (xj.bottomRows(nrows).rowwise() - xj.row(i)).matrix().rowwise().norm();
			k += nrows;
		}
		
		// Calculate distance metric normalization constants.
		f_type r0 = 0.35, n = 8, m = 12; 

		dxi.array() = (1. - (dxi.array()/r0).pow(n))/(1. - (dxi.array()/r0).pow(m));
		dxj.array() = (1. - (dxj.array()/r0).pow(n))/(1. - (dxj.array()/r0).pow(m));

		f_type norm = std::sqrt(dxi.sum()*dxj.sum());

		return std::sqrt(1.0/norm*(dxi - dxj).array().square().sum());
	}
}