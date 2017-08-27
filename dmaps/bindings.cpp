#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include "distance_matrix.h"
#include "metrics.h"

namespace py = pybind11;

PYBIND11_MODULE(dmaps, m)
{
	py::class_<dmaps::distance_matrix>(m, "DistanceMatrix")
        .def(py::init<dmaps::matrix_t, int>(), 
            py::arg(), py::arg("num_threads") = 0)
        .def(py::init<dmaps::matrix_t, dmaps::vector_t, int>(),
            py::arg(), py::arg("weights"), py::arg("num_threads") = 0
        )
		.def("get_coordinates", &dmaps::distance_matrix::get_coordinates)
		.def("get_distances", &dmaps::distance_matrix::get_distances)
		.def("compute", &dmaps::distance_matrix::compute,
			py::arg("metric") = py::cpp_function(&dmaps::rmsd)
		);
	
	py::module m2 = m.def_submodule("metrics");
	m2.def("rmsd", &dmaps::rmsd);
}