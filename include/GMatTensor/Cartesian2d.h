/**
\file
\copyright Copyright 2020. Tom de Geus. All rights reserved.
\license This project is released under the MIT License.
*/

#ifndef GMATTENSOR_CARTESIAN2D_H
#define GMATTENSOR_CARTESIAN2D_H

#include <xtensor/xadapt.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "config.h"
#include "version.h"

namespace GMatTensor {

/**
Tensors and tensor operations for a(n) array of 2d tensors of different rank,
defined in a Cartesian coordinate system.
*/
namespace Cartesian2d {

/**
Random 2nd-order tensor (for example for use in testing).

\return [2, 2] array.
*/
inline array_type::tensor<double, 2> Random2();

/**
Random 4th-order tensor (for example for use in testing).

\return [2, 2, 2, 2] array.
*/
inline array_type::tensor<double, 4> Random4();

/**
2nd-order null tensor (all components equal to zero).

\return [2, 2] array.
*/
inline array_type::tensor<double, 2> O2();

/**
4th-order null tensor (all components equal to zero).

\return [2, 2, 2, 2] array.
*/
inline array_type::tensor<double, 4> O4();

/**
2nd-order identity tensor.
By definition

\f$ I_{ij} = \delta_{ij} \f$

such that

\f$ I \cdot A = A \f$

or in index notation

\f$ I_{ij} A_{jk} = A_{ik} \f$

See A2_dot_B2().

\return [2, 2] array.
*/
inline array_type::tensor<double, 2> I2();

/**
Result of the dyadic product of two 2nd-order identity tensors (see I2()).
By definition

\f$ (II)_{ijkl} = \delta_{ij} \delta_{kl} \f$

such that

\f$ II : A = tr(A) I \f$

or in index notation

\f$ (II)_{ijkl} A_{lk} = tr(A) I_{ij} \f$

See A4_ddot_B2(), Trace(), I2().

\return [2, 2, 2, 2] array.
*/
inline array_type::tensor<double, 4> II();

/**
Fourth order unit tensor.
By definition

\f$ I_{ijkl} = \delta_{il} \delta_{jk} \f$

such that

\f$ I : A = A \f$

or in index notation

\f$ I_{ijkl} A_{lk} = A_{ij} \f$

See A4_ddot_B2().

\return [2, 2, 2, 2] array.
*/
inline array_type::tensor<double, 4> I4();

/**
Right-transposed fourth order unit tensor.
By definition

\f$ I_{ijkl} = \delta_{ik} \delta_{jl} \f$

such that

\f$ I : A = A^T \f$

or in index notation

\f$ I_{ijkl} A_{lk} = A_{ji} \f$

See A4_ddot_B2().

\return [2, 2, 2, 2] array.
*/
inline array_type::tensor<double, 4> I4rt();

/**
Fourth order symmetric projection.
By definition

    I = 0.5 * (I4() + I4rt())

such that

 \f$ I : A = sym(A) \f$

 or in index notation

 \f$ I_{ijkl} A_{lk} = (A_{ij} + A_{ji}) / 2 \f$

See A4_ddot_B2(), Sym().

\return [2, 2, 2, 2] array.
*/
inline array_type::tensor<double, 4> I4s();

/**
Fourth order deviatoric projection.
By definition

    I = I4s() - 0.5 * II()

such that

\f$ I : A = sym(A) - tr(A) / 2 \f$

See A4_ddot_B2(), Deviatoric().

\return [2, 2, 2, 2] array.
*/
inline array_type::tensor<double, 4> I4d();

/**
Trace or 2nd-order tensor.

\f$ tr(A) = A_{ii} \f$

To write to allocated data use trace().

\param A [..., 2, 2] array.
\return [...] array.
*/
template <class T>
inline auto Trace(const T& A);

/**
Same as Trace() but writes to externally allocated output.

\param A [..., 2, 2] array.
\param ret output [...] array.
*/
template <class T, class R>
inline void trace(const T& A, R& ret);

/**
Hydrostatic part of a tensor

    == trace(A) / 2 == trace(A) / d

where ``d = 2``.
To write to allocated output use hydrostatic().

\param A [..., 2, 2] array.
\return [...] array.
*/
template <class T>
inline auto Hydrostatic(const T& A);

/**
Same as Hydrostatic() but writes to externally allocated output.

\param A [..., 2, 2] array.
\param ret output [...] array.
*/
template <class T, class R>
inline void hydrostatic(const T& A, R& ret);

/**
Double tensor contraction

\f$ c = A : B \f$

or in index notation

\f$ c = A_{ij} A_{ji} \f$

To write to allocated data use A2_ddot_B2(const T& A, const T& B, R& ret).

\param A [..., 2, 2] array.
\param B [..., 2, 2] array.
\return [...] array.
*/
template <class T>
inline auto A2_ddot_B2(const T& A, const T& B);

/**
Same as A2_ddot_B2(const T& A, const T& B) but writes to externally allocated output.

\param A [..., 2, 2] array.
\param B [..., 2, 2] array.
\param ret output [...] array.
*/
template <class T, class R>
inline void A2_ddot_B2(const T& A, const T& B, R& ret);

/**
Same as A2_ddot_B2(const T& A, const T& B, R& ret) but for symmetric tensors.
This function is slightly faster.
There is no assertion to check the symmetry.
To write to allocated data use A2s_ddot_B2s(const T& A, const T& B, R& ret).

\param A [..., 2, 2] array.
\param B [..., 2, 2] array.
\return [...] array.
*/
template <class T>
inline auto A2s_ddot_B2s(const T& A, const T& B);

/**
Same as A2s_ddot_B2s(const T& A, const T& B) but writes to externally allocated output.

\param A [..., 2, 2] array.
\param B [..., 2, 2] array.
\param ret output [...] array.
*/
template <class T, class R>
inline void A2s_ddot_B2s(const T& A, const T& B, R& ret);

/**
Norm of the tensor's deviator:

\f$ \sqrt{(dev(A))_{ij} (dev(A))_{ji}} \f$

To write to allocated data use norm_deviatoric().

\param A [..., 2, 2] array.
\return [...] array.
*/
template <class T>
inline auto Norm_deviatoric(const T& A);

/**
Same as Norm_deviatoric() but writes to externally allocated output.

\param A [..., 2, 2] array.
\param ret output [...] array
*/
template <class T, class R>
inline void norm_deviatoric(const T& A, R& ret);

/**
Deviatoric part of a tensor:

    A - Hydrostatic(A) * I2

See Hydrostatic().
To write to allocated data use deviatoric().

\param A [..., 2, 2] array.
\return [..., 2, 2] array.
*/
template <class T>
inline auto Deviatoric(const T& A);

/**
Same as Deviatoric() but writes to externally allocated output.

\param A [..., 2, 2] array.
\param ret output [..., 2, 2] array.
*/
template <class T, class R>
inline void deviatoric(const T& A, R& ret);

/**
Symmetric part of a tensor:

\f$ (A + A^T) / 2 \f$

of in index notation

\f$ (A_{ij} + A_{ji}) / 2 \f$

To write to allocated data use sym().

\param A [..., 2, 2] array.
\return [..., 2, 2] array.
*/
template <class T>
inline auto Sym(const T& A);

/**
Same as Sym() but writes to externally allocated output.

\param A [..., 2, 2] array.
\param ret output [..., 2, 2] array, may be the same reference as ``A``.
*/
template <class T, class R>
inline void sym(const T& A, R& ret);

/**
Dot-product (single tensor contraction)

\f$ C = A \cdot B \f$

or in index notation

\f$ C_{ik} = A_{ij} B_{jk} \f$

To write to allocated data use A2_dot_B2(const T& A, const T& B, R& ret).

\param A [..., 2, 2] array.
\param B [..., 2, 2] array.
\return [..., 2, 2] array.
*/
template <class T>
inline auto A2_dot_B2(const T& A, const T& B);

/**
Same as A2_dot_B2(const T& A, const T& B) but writes to externally allocated output.

\param A [..., 2, 2] array.
\param B [..., 2, 2] array.
\param ret output [..., 2, 2] array.
*/
template <class T, class R>
inline void A2_dot_B2(const T& A, const T& B, R& ret);

/**
Dyadic product

\f$ C = A \otimes B \f$

or in index notation

\f$ C_{ijkl} = A_{ij} B_{kl} \f$

To write to allocated data use A2_dyadic_B2(const T& A, const T& B, R& ret).

\param A [..., 2, 2] array.
\param B [..., 2, 2] array.
\return [..., 2, 2, 2, 2] array.
*/
template <class T>
inline auto A2_dyadic_B2(const T& A, const T& B);

/**
Same as A2_dyadic_B2(const T& A, const T& B) but writes to externally allocated output.

\param A [..., 2, 2] array.
\param B [..., 2, 2] array.
\param ret output [..., 2, 2, 2, 2] array.
*/
template <class T, class R>
inline void A2_dyadic_B2(const T& A, const T& B, R& ret);

/**
Double tensor contraction

\f$ C = A : B \f$

or in index notation

\f$ C_{ij} = A_{ijkl} A_{lk} \f$

To write to allocated data use A4_ddot_B2(const T& A, const U& B, R& ret).

\param A [..., 2, 2, 2, 2] array.
\param B [..., 2, 2] array.
\return [..., 2, 2] array.
*/
template <class T, class U>
inline auto A4_ddot_B2(const T& A, const U& B);

/**
Same as A4_ddot_B2(const T& A, const U& B) but writes to externally allocated output.

\param A [..., 2, 2, 2, 2] array.
\param B [..., 2, 2] array.
\param ret output [..., 2, 2] array.
*/
template <class T, class U, class R>
inline void A4_ddot_B2(const T& A, const U& B, R& ret);

/**
Size of the underlying array.

\param A [..., 2, 2] array.
\return `prod([...])`.
*/
template <class T>
inline auto underlying_size_A2(const T& A);

/**
Size of the underlying array.

\param A [..., 2, 2] array.
\return `prod([...])`.
*/
template <class T>
inline auto underlying_size_A4(const T& A);

/**
Shape of the underlying array.

\param A [..., 2, 2] array.
\return `[...]`.
*/
template <class T>
inline auto underlying_shape_A2(const T& A);

/**
Shape of the underlying array.

\param A [..., 2, 2] array.
\return `[...]`.
*/
template <class T>
inline auto underlying_shape_A4(const T& A);

/**
Array of tensors:
-   scalars: shape ``[...]``.
-   2nd-order tensors: shape ``[..., 2, 2]``.
-   4nd-order tensors: shape ``[..., 2, 2, 2, 2]``.

\tparam N The rank of the array (the actual rank is increased with the tensor-rank).
*/
template <size_t N>
class Array {
public:
    /**
    Rank of the array (the actual rank is increased with the tensor-rank).
    */
    constexpr static std::size_t rank = N;

    Array() = default;

    /**
    Constructor.

    \param shape The shape of the array (or scalars).
    */
    Array(const std::array<size_t, N>& shape);

    /**
    Shape of the array (of scalars).

    \return List of size #rank.
    */
    const std::array<size_t, N>& shape() const;

    /**
    Shape of the array of second-order tensors.

    \return List of size #rank + 2.
    */
    const std::array<size_t, N + 2>& shape_tensor2() const;

    /**
    Shape of the array of fourth-order tensors.

    \return List of size #rank + 4.
    */
    const std::array<size_t, N + 4>& shape_tensor4() const;

    /**
    Array of Cartesian2d::O2()

    \return [shape(), 2, 2]
    */
    array_type::tensor<double, N + 2> O2() const;

    /**
    Array of Cartesian2d::O4()

    \return [shape(), 2, 2, 2, 2]
    */
    array_type::tensor<double, N + 4> O4() const;

    /**
    Array of Cartesian2d::I2()

    \return [shape(), 2, 2]
    */
    array_type::tensor<double, N + 2> I2() const;

    /**
    Array of Cartesian2d::II()

    \return [shape(), 2, 2, 2, 2]
    */
    array_type::tensor<double, N + 4> II() const;

    /**
    Array of Cartesian2d::I4()

    \return [shape(), 2, 2, 2, 2]
    */
    array_type::tensor<double, N + 4> I4() const;

    /**
    Array of Cartesian2d::I4rt()

    \return [shape(), 2, 2, 2, 2]
    */
    array_type::tensor<double, N + 4> I4rt() const;

    /**
    Array of Cartesian2d::I4s()

    \return [shape(), 2, 2, 2, 2]
    */
    array_type::tensor<double, N + 4> I4s() const;

    /**
    Array of Cartesian2d::I4d()

    \return [shape(), 2, 2, 2, 2]
    */
    array_type::tensor<double, N + 4> I4d() const;

protected:
    /**
    Constructor 'alias'. Can be used by constructor of derived classes.

    \param shape The shape of the array (or scalars).
    */
    void init(const std::array<size_t, N>& shape);

    /** Number of dimensions of tensors. */
    static constexpr size_t m_ndim = 2;

    /** Storage stride for 2nd-order tensors (\f$ 2^2 \f$). */
    static constexpr size_t m_stride_tensor2 = 4;

    /** Storage stride for 4th-order tensors (\f$ 2^4 \f$). */
    static constexpr size_t m_stride_tensor4 = 16;

    /** Size of the array (of scalars) == prod(#m_shape). */
    size_t m_size;

    /** Shape of the array (of scalars). */
    std::array<size_t, N> m_shape;

    /** Shape of an array of 2nd-order tensors == [#m_shape, 2, 2]. */
    std::array<size_t, N + 2> m_shape_tensor2;

    /** Shape of an array of 4th-order tensors == [#m_shape, 2, 2, 2, 2]. */
    std::array<size_t, N + 4> m_shape_tensor4;
};

/**
API for individual tensors with pointer-only input.
No arrays of tensors are allowed, hence the input is fixed to:

-   Second order tensors, ``size = 2 * 2 = 4``.
    Storage convention ``(xx, xy, yx, yy)``.

-   Fourth order tensors, ``size = 2 * 2 * 2 * 2 = 16``.
*/
namespace pointer {

/**
See Cartesian2d::O2()

\param ret output 2nd order tensor
*/
template <class T>
inline void O2(T* ret);

/**
See Cartesian2d::O4()

\param ret output 2nd order tensor
*/
template <class T>
inline void O4(T* ret);

/**
See Cartesian2d::I2()

\param ret output 2nd order tensor
*/
template <class T>
inline void I2(T* ret);

/**
See Cartesian2d::II()

\param ret output 2nd order tensor
*/
template <class T>
inline void II(T* ret);

/**
See Cartesian2d::I4()

\param ret output 2nd order tensor
*/
template <class T>
inline void I4(T* ret);

/**
See Cartesian2d::I4rt()

\param ret output 2nd order tensor
*/
template <class T>
inline void I4rt(T* ret);

/**
See Cartesian2d::I4s()

\param ret output 2nd order tensor
*/
template <class T>
inline void I4s(T* ret);

/**
See Cartesian2d::I4d()

\param ret output 2nd order tensor
*/
template <class T>
inline void I4d(T* ret);

/**
See Cartesian2d::Trace()

\param A 2nd order tensor
\return scalar
*/
template <class T>
inline T Trace(const T* A);

/**
See Cartesian2d::Hydrostatic()

\param A 2nd order tensor
\return scalar
*/
template <class T>
inline T Hydrostatic(const T* A);

/**
See Cartesian2d::Sym()

\param A 2nd order tensor
\param ret 2nd order tensor, may be the same pointer as ``A``
*/
template <class T>
inline void sym(const T* A, T* ret);

/**
Returns Cartesian2d::Hydrostatic() and computes Cartesian2d::Deviatoric()

\param A 2nd order tensor
\param ret 2nd order tensor, may be the same pointer as ``A``
\return scalar
*/
template <class T>
inline T Hydrostatic_deviatoric(const T* A, T* ret);

/**
Double tensor contraction of the tensor's deviator

\f$ (dev(A))_{ij} (dev(A))_{ji} \f$

\param A 2nd order tensor
\return scalar
*/
template <class T>
inline T Deviatoric_ddot_deviatoric(const T* A);

/**
See Cartesian2d::Norm_deviatoric()

\param A 2nd order tensor
\return scalar
*/
template <class T>
inline T Norm_deviatoric(const T* A);

/**
See Cartesian2d::A2_ddot_B2()

\param A 2nd order tensor
\param B 2nd order tensor
\return scalar
*/
template <class T>
inline T A2_ddot_B2(const T* A, const T* B);

/**
See Cartesian2d::A2s_ddot_B2s()

\param A 2nd order tensor
\param B 2nd order tensor
\return scalar
*/
template <class T>
inline T A2s_ddot_B2s(const T* A, const T* B);

/**
See Cartesian2d::A2_dyadic_B2()

\param A 2nd order tensor
\param B 2nd order tensor
\param ret output 4th order tensor
*/
template <class T>
inline void A2_dyadic_B2(const T* A, const T* B, T* ret);

/**
See Cartesian2d::A2_dot_B2()

\param A 2nd order tensor
\param B 2nd order tensor
\param ret output 2th order tensor
*/
template <class T>
inline void A2_dot_B2(const T* A, const T* B, T* ret);

/**
See Cartesian2d::A4_ddot_B2()

\param A 4th order tensor
\param B 2nd order tensor
\param ret output 2th order tensor
*/
template <class T>
inline void A4_ddot_B2(const T* A, const T* B, T* ret);

} // namespace pointer

} // namespace Cartesian2d
} // namespace GMatTensor

#include "Cartesian2d.hpp"
#include "Cartesian2d_Array.hpp"

#endif
