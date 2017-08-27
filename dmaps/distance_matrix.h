#pragma once 

#define f_type double

#include <Eigen/Core>

namespace dmaps
{
	using matrix_t = Eigen::Matrix<f_type, Eigen::Dynamic, Eigen::Dynamic>; 
	using matrix3_t = Eigen::Matrix<f_type, Eigen::Dynamic, 3>;
	
	class distance_matrix
	{
	private:
		// Distance matrix.
		matrix_t d_;

		// Coordinates.
		matrix_t x_;
	
	public:
		distance_matrix(const matrix_t& x) : 
		x_(x)
		{}
		
		const matrix_t& get_coordinates(); 

		void compute();
	};
}