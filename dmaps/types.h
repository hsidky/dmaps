#pragma once 

#include <Eigen/Core>

#define f_type double

namespace dmaps
{
	using matrix_t = Eigen::Matrix<f_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; 
	using vector_t = Eigen::Matrix<f_type, Eigen::Dynamic, 1>;
	using vector3_t = Eigen::Matrix<f_type, 3, 1>;
	using matrix3_t = Eigen::Matrix<f_type, Eigen::Dynamic, 3, Eigen::RowMajor>;
	using matrix33_t = Eigen::Matrix<f_type, 3, 3, Eigen::RowMajor>;
}