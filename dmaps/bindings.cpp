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
		.def("compute", (void(distance_matrix::*)(const std::function<f_type(const vector_t&, const vector_t&, const vector_t&)>&)) &distance_matrix::compute,
			py::arg("metric") = py::cpp_function(&rmsd)
		)
        .def("compute_single", (vector_t(distance_matrix::*)(const vector_t&, const std::function<f_type(const vector_t&, const vector_t&, const vector_t&)>&)) &distance_matrix::compute,
			py::arg("coordinate"),
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
        .def("get_kernel_bandwidth", &diffusion_map::get_kernel_bandwidth)
        .def("sum_similarity_matrix", &diffusion_map::sum_similarity_matrix,
        py::arg("epsilon"), py::arg("alpha") = 1.0)
        .def("compute", &diffusion_map::compute,
            py::arg("n") = 0, py::arg("alpha") = 1.0, py::arg("beta") = 0.0)
        .def("get_eigenvectors", &diffusion_map::get_eigenvectors)
        .def("get_eigenvalues", &diffusion_map::get_eigenvalues)
        .def("nystrom", &diffusion_map::nystrom, 
            py::arg("distances"), py::arg("alpha") = 1.0, py::arg("beta") = 0.0)
        .def("get_kernel_matrix", &diffusion_map::get_kernel_matrix);
	
	// Metrics submodule.
	py::module m2 = m.def_submodule("metrics");
    m2.def("rmsd", &rmsd);
    m2.def("euclidean", &euclidean);
    m2.def("contact_map", &contact_map);
}