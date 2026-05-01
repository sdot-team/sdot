#pragma once

#include "common_macros.h"
#include "DsVec.h"

#include <ostream>

namespace sdot {
template<class T,int ct_rank,class Arch> class Tensor;
template<class T,int ct_size,class Arch> class DsVec;

/// view on strided data (strides in bytes, handles non-contiguous arrays)
template<class T,int ct_rank,class Arch>
class TensorView {
public:
    using          value_type        = T;
    using          Strides           = DsVec<SI,ct_rank,Arch>; ///< byte strides
    using          Sizes             = DsVec<PI,ct_rank,Arch>;
    using          Ptr               = std::conditional_t<std::is_const_v<T>,const std::byte*,std::byte*>;

    static HD auto make_invalid      ( PI rank = ct_rank ); ///< invalid TensorView — is_valid()==false, _ptr==&_sentinel

    HD             TensorView        ( T *data, Sizes sizes, Strides strides );
    HD             TensorView        ( T *data, Sizes sizes );
    HD             TensorView        ( T *data, PI size );
    HD             TensorView        ( Rank, PI rank );


    HD T&          operator()        ( const auto &indices, auto ...rem ) const requires ( requires { indices.size(); } );
    HD T&          operator()        ( PI index, auto ...rem ) const;
    HD T&          operator()        () const;

    HD T&          operator[]        ( const auto &index ) const;

    HD auto        partial           ( auto ...indices ) const;

    HD TensorView& get_data_from     ( const TensorView<T,ct_rank,Arch> &that );
    void           fill_with         ( T value );

    static HD auto contiguous_strides( const Sizes &ext ) -> Strides;
    HD Strides     strides           () const;
    HD SI          stride            ( PI d ) const;

    HD PI          total_size        () const;
    HD bool        is_invalid        () const; ///<
    HD bool        is_valid          () const; ///< false iff constructed from None/nullopt (_ptr == nullptr)
    HD bool        empty             () const;
    Sizes          sizes             () const;
    HD PI          size              ( PI d ) const;
    HD PI          size              () const;

    HD PI          rank              () const;

    HD T*          data              () const;

    HD auto        begin             () const;
    HD auto        end               () const;

    HD void        for_each_index    ( auto &&func, PI sub = 0 ) const;

    HD bool        is_contiguous     () const; ///< true iff strides match row-major contiguous layout
    HD auto        unsqueeze         () const; ///< append a trailing dimension of size 1 (preserves strides)

    T_U auto       sum_along_axis_1  () const -> Tensor<U,1,Arch>;
    void           with_cpu_version  ( auto &&func ) const;
    auto           squeeze           ( PI axis, PI index = 0 ) const;
    HD auto        row               ( PI index ) const;

    static std::byte   _sentinel;    ///< address used as invalid marker — never points to real data

    friend std::ostream &operator<<( std::ostream &os, const TensorView &p ) {
        if constexpr( ct_rank == 0 )
            return os << p();
        else if constexpr ( ct_rank == 1 ) {
            for( sdot::PI i = 0; i < p.size(); ++i )
                os << ( i ? ", " : "" ) << p[ i ];
            return os;
        } else {
            for( sdot::PI i = 0; i < p.size( 0 ); ++i )
                os << "\n" << p.row( i );
            return os;
        }
    }

private:
    Strides        _strides;         ///< byte strides
    Sizes          _sizes;           ///<
    Ptr            _ptr;             ///<
};

} // namespace sdot

#ifdef __CUDACC__
template<class T,int ct_rank>
std::ostream &operator<<( std::ostream &os, const sdot::TensorView<T,ct_rank,sdot::Cuda> &p );
#endif

#include "TensorView.cxx" // IWYU pragma: export
