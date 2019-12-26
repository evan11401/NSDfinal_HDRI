#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

#include <mkl.h>

#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <sstream>
#include <functional>

namespace py = pybind11;
using namespace std;

class Matrix {

public:

    Matrix(size_t nrow, size_t ncol)
      : m_nrow(nrow), m_ncol(ncol)
    {
        reset_buffer(nrow, ncol);
    }

    Matrix(size_t nrow, size_t ncol, vector<double> const & vec)
      : m_nrow(nrow), m_ncol(ncol)
    {
        reset_buffer(nrow, ncol);
        (*this) = vec;
    }

    Matrix & operator=(vector<double> const & vec)
    {
        if (size() != vec.size())
        {
            throw out_of_range("number of elements mismatch");
        }

        size_t k = 0;
        for (size_t i=0; i<m_nrow; ++i)
        {
            for (size_t j=0; j<m_ncol; ++j)
            {
                (*this)(i,j) = vec[k];
                ++k;
            }
        }

        return *this;
    }

    Matrix(Matrix const & other)
      : m_nrow(other.m_nrow), m_ncol(other.m_ncol)
    {
        reset_buffer(other.m_nrow, other.m_ncol);
        for (size_t i=0; i<m_nrow; ++i)
        {
            for (size_t j=0; j<m_ncol; ++j)
            {
                (*this)(i,j) = other(i,j);
            }
        }
    }

    Matrix & operator=(Matrix const & other)
    {
        if (this == &other) { return *this; }
        if (m_nrow != other.m_nrow || m_ncol != other.m_ncol)
        {
            reset_buffer(other.m_nrow, other.m_ncol);
        }
        for (size_t i=0; i<m_nrow; ++i)
        {
            for (size_t j=0; j<m_ncol; ++j)
            {
                (*this)(i,j) = other(i,j);
            }
        }
        return *this;
    }

    Matrix(Matrix && other)
      : m_nrow(other.m_nrow), m_ncol(other.m_ncol)
    {
        reset_buffer(0, 0);
        swap(m_nrow, other.m_nrow);
        swap(m_ncol, other.m_ncol);
        swap(m_buffer, other.m_buffer);
    }

    Matrix & operator=(Matrix && other)
    {
        if (this == &other) { return *this; }
        reset_buffer(0, 0);
        swap(m_nrow, other.m_nrow);
        swap(m_ncol, other.m_ncol);
        swap(m_buffer, other.m_buffer);
        return *this;
    }

    ~Matrix()
    {
        reset_buffer(0, 0);
    }

    double   operator() (size_t row, size_t col) const { return m_buffer[index(row, col)]; }
    double & operator() (size_t row, size_t col)       { return m_buffer[index(row, col)]; }

    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }

    size_t size() const { return m_nrow * m_ncol; }
    double buffer(size_t i) const { return m_buffer[i]; }
    vector<double> buffer_vector() const { return vector<double>(m_buffer, m_buffer+size()); }
    double * get_buffer() const {return m_buffer;}
    py::array_t<double> numpy() const
    {
        auto capsule = py::capsule(m_buffer);
        return py::array_t<double>({m_nrow, m_ncol}, m_buffer, capsule);
    }
private:

    size_t index(size_t row, size_t col) const
    {
        return row + col * m_nrow;
    }

    void reset_buffer(size_t nrow, size_t ncol)
    {
        if (m_buffer) { delete[] m_buffer; }
        const size_t nelement = nrow * ncol;
        if (nelement) { m_buffer = new double[nelement](); }
        else          { m_buffer = nullptr; }
        m_nrow = nrow;
        m_ncol = ncol;
    }

    size_t m_nrow = 0;
    size_t m_ncol = 0;
    double * m_buffer = nullptr;

};
bool operator== (Matrix const & mat1, Matrix const & mat2)
{
    if ((mat1.ncol() != mat2.ncol()) && (mat1.nrow() != mat2.ncol()))
    {
        return false;
    }

    for (size_t i=0; i<mat1.nrow(); ++i)
    {
        for (size_t j=0; j<mat1.ncol(); ++j)
        {
            if (mat1(i, j) != mat2(i, j))
            {
                return false;
            }
        }
    }

    return true;
}
bool operator!= (Matrix const & mat1, Matrix const & mat2)
{
    return !(mat1 == mat2);
}
void validate_multiplication(Matrix const & mat1, Matrix const & mat2)
{
    if (mat1.ncol() != mat2.nrow())
        {
	        throw std::out_of_range(
	            "the number of first matrix column "
		    "differs from that of second matrix row");
	}
}
/*
 * Naive matrix matrix multiplication.
 */
Matrix multiply_naive(Matrix const & mat1, Matrix const & mat2)
{
    validate_multiplication(mat1, mat2);
    Matrix ret(mat1.nrow(), mat2.ncol());
    for (size_t i=0; i<ret.nrow(); ++i)
    {
        for (size_t k=0; k<ret.ncol(); ++k)
        {
            for (size_t j=0; j<mat1.ncol(); ++j)
            {
                ret(i, k) += mat1(i,j) * mat2(j,k);
            }
        }
    }

    return ret;
}

Matrix multiply_mkl(Matrix const & mat1, Matrix const & mat2)
{
    validate_multiplication(mat1, mat2);
    Matrix ret(mat1.nrow(), mat2.ncol());
    cblas_dgemm(
        CblasRowMajor /* const CBLAS_LAYOUT Layout */
      , CblasNoTrans /* const CBLAS_TRANSPOSE transa */
      , CblasNoTrans /* const CBLAS_TRANSPOSE transb */
      , mat1.nrow() /* const MKL_INT m */
      , mat2.ncol() /* const MKL_INT n */
      , mat1.ncol() /* const MKL_INT k */
      , 1.0 /* const double alpha */
      , mat1.get_buffer() /* const double *a */
      , mat1.ncol() /* const MKL_INT lda */
      , mat2.get_buffer() /* const double *b */
      , mat2.ncol() /* const MKL_INT ldb */
      , 0.0 /* const double beta */
      , ret.get_buffer() /* double * c */
      , ret.ncol() /* const MKL_INT ldc */
    );

    return ret;
}

Matrix multiply_tile(Matrix const & mat1, Matrix const & mat2, size_t tsize){
    validate_multiplication(mat1, mat2);
    Matrix ret(mat1.nrow(), mat2.ncol());
    size_t a = mat1.nrow(),b = mat2.ncol(), c = mat1.ncol();
    size_t s = tsize*2;
    //s = pow(2,int(log2(s)));
    for(size_t i=0;i<a;i+=s)
    	for(size_t j=0;j<b;j+=s)
	    for(size_t k=0;k<c;k+=s){
	    	const size_t mini = min(i+s,a), minj = min(j+s,b), mink = min(k+s,c);
	        for(size_t ii=i;ii<mini;++ii){
		       for(size_t jj=j;jj<minj;++jj){
		           double sum=0;
		           for(size_t kk=k;kk<mink;++kk){
			       sum+=mat1(ii,kk)*mat2(kk,jj);
			   }
			   ret(ii,jj) +=sum;
		       }
		}
	    }
    return ret;
}


PYBIND11_MODULE(_matrix, m) {
	m.doc() = "Matrix";
	m.def("multiply_naive", &multiply_naive,
	          "naive multiply matrices.");
	m.def("multiply_mkl", &multiply_mkl,
		  "mkl multiply matrices.");
	m.def("multiply_tile", &multiply_tile);
	py::class_<Matrix>(m, "Matrix")
	        .def(py::init<int, int>())
		.def_property("array", &Matrix::numpy, nullptr)
		.def_property_readonly("nrow", &Matrix::nrow)
		.def_property_readonly("ncol", &Matrix::ncol)
		.def("__setitem__", [](Matrix &M,
		std::pair<size_t, size_t> const id, double const &val){
		            M(id.first, id.second) = val;
		})
		.def("__getitem__", [](Matrix const &M, std::pair<size_t, size_t> const id){
		            return M(id.first, id.second);
		})
		.def("__eq__", [](Matrix &A, Matrix const &B){
		            return A == B;
		});

}