#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <cmath>
#include <iostream>
#include "distance_matrix.h"

namespace py = pybind11;
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
		return sqrt((w.asDiagonal()*(xi - xj*R).array().square().matrix()).sum()/wtot);
	}

	const matrix_t& distance_matrix::get_coordinates()
	{
		return x_;
	}

	const matrix_t& distance_matrix::get_distances()
	{
		return d_;
	}

	void distance_matrix::compute()
	{
		d_ = matrix_t::Zero(x_.rows(), x_.rows());
		for(int i = 0; i < d_.rows() - 1; ++i)
		{
			for(int j = i + 1; j < d_.rows(); ++j)
			{
				d_(i,j) = rmsd(x_.row(i), x_.row(j), w_);
				d_(j,i) = d_(i,j);
			}
		}
	}
}

PYBIND11_MODULE(dmaps, m)
{
	py::class_<dmaps::distance_matrix>(m, "DistanceMatrix")
		.def(py::init<dmaps::matrix_t>())
		.def(py::init<dmaps::matrix_t, dmaps::vector_t>())
		.def("get_coordinates", &dmaps::distance_matrix::get_coordinates)
		.def("get_distances", &dmaps::distance_matrix::get_distances)
		.def("compute", &dmaps::distance_matrix::compute);
}