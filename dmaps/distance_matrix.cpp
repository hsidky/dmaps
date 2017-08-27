#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "distance_matrix.h"

namespace py = pybind11;

namespace dmaps
{
	const matrix_t& distance_matrix::get_coordinates()
	{
		return x_;
	}

	void distance_matrix::compute()
	{
		d_ = matrix_t::Zero(x_.rows(), x_.rows());
		for(int i = 0; i < d_.rows() - 1; ++i)
			for(int j = i+1; j < d_.rows(); ++j)
			{
				
			}
	}
}

PYBIND11_MODULE(dmaps, m)
{
	py::class_<dmaps::distance_matrix>(m, "DistanceMatrix")
		.def(py::init<dmaps::matrix_t>())
		.def("get_coordinates", &dmaps::distance_matrix::get_coordinates);
}