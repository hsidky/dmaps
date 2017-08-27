#pragma once 

#define f_type double

#include <stdexcept>
#include <Eigen/Core>

namespace dmaps
{
	using matrix_t = Eigen::Matrix<f_type, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; 
	using vector_t = Eigen::Matrix<f_type, Eigen::Dynamic, 1>;
	using vector3_t = Eigen::Matrix<f_type, 3, 1>;
	using matrix3_t = Eigen::Matrix<f_type, Eigen::Dynamic, 3, Eigen::RowMajor>;
	using matrix33_t = Eigen::Matrix<f_type, 3, 3>;
	
	class distance_matrix
	{
	private:
		// Distance matrix.
		matrix_t d_;

		// Weights vector. 
		vector_t w_;

		// Coordinates.
		matrix_t x_;
	
	public:
		distance_matrix(const matrix_t& x) : 
		x_(x)
		{
			w_ = vector_t::Ones(x_.cols()/3);
		}

		distance_matrix(const matrix_t& x, const vector_t& w) : 
		x_(x), w_(w)
		{
			if(x_.cols()/3 != w.size())
				throw std::runtime_error("Length of weights vector must match dimension of coordinates.");
		}
				
		const matrix_t& get_coordinates(); 

		const matrix_t& get_distances();

		void compute();
	};
}