#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include "distance_matrix.h"
#include "diffusion_map.h"
#include "metrics.h"

namespace py = pybind11;
using namespace dmaps;

PYBIND11_MODULE(dmaps, m)
{
	// Distance matrix class.
	py::class_<distance_matrix>(m, "DistanceMatrix")
        .def(py::init<const matrix_t&, int>(), 
            py::arg(), py::arg("num_threads") = 0)
        .def(py::init<const matrix_t&, const vector_t&, int>(),
            py::arg(), py::arg("weights"), py::arg("num_threads") = 0
		)
		.def(py::init<const std::string&, int>(),
			py::arg(), py::arg("num_threads") = 0
		)
		.def("get_coordinates", &distance_matrix::get_coordinates)
		.def("get_distances", &distance_matrix::get_distances)
		.def("compute", &distance_matrix::compute,
			py::arg("metric") = py::cpp_function(&rmsd)
		)
		.def("save", &distance_matrix::save);
	
	// Diffusion map class.
	py::class_<diffusion_map>(m, "DiffusionMap")
        .def(py::init<const matrix_t&, const vector_t&, int>(),
            py::arg(), py::arg("weights") = vector_t(), py::arg("num_threads") = 0
        )
        .def(py::init<const distance_matrix&, const vector_t&, int>(),
            py::arg(), py::arg("weights") = vector_t(), py::arg("num_threads") = 0
        )
        .def("set_kernel_bandwidth", (void(diffusion_map::*)(f_type)) &diffusion_map::set_kernel_bandwidth)
        .def("set_kernel_bandwidth", (void(diffusion_map::*)(const vector_t&)) &diffusion_map::set_kernel_bandwidth)
        .def("get_kernel_bandwidth", &diffusion_map::get_kernel_bandwidth)
        .def("sum_similarity_matrix", &diffusion_map::sum_similarity_matrix)
        .def("estimate_local_scale", &diffusion_map::estimate_local_scale,
            py::arg("k") = 0)
        .def("compute", &diffusion_map::compute)
        .def("get_eigenvectors", &diffusion_map::get_eigenvectors)
        .def("get_eigenvalues", &diffusion_map::get_eigenvalues)
        .def("get_kernel_matrix", &diffusion_map::get_kernel_matrix);
	
	// Metrics submodule.
	py::module m2 = m.def_submodule("metrics");
	m2.def("rmsd", &rmsd);
}