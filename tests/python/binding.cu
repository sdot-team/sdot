#include <sdot/jax_ffi_wrappers.h>
#include <nanobind/nanobind.h>

namespace nb = nanobind;
using namespace sdot;

template<typename TF, typename MemorySpace, typename TI>
struct Parameters {
    HD auto batch_sizes() const { return tuple(  ); }
    HD auto operator()( Tuple<> ) const { return *this; }
    TensorView<TF,Tuple<>,DECAYED_TYPE_OF( contiguous_strides<TF>( Tuple<>() ) ),MemorySpace> output;
    TensorView<FP64,Tuple<TI>,DECAYED_TYPE_OF( contiguous_strides<FP64>( Tuple<TI>() ) ),MemorySpace> input;
};

xla::ffi::Error impl_using_P_DECAYED_TYPE_OF_p_run_sequentiaiuW42OoCcWd( xla::ffi::Buffer<xla::ffi::F64> di0, xla::ffi::Buffer<xla::ffi::U8> u8_input_buffer, xla::ffi::ResultBuffer<xla::ffi::F64> o0, xla::ffi::ResultBuffer<xla::ffi::U64> u64_output_buffer, cudaStream_t stream ) {
    using TI = SI64;
    ExecutionContext_Cuda::default_stream = stream;
    ExecutionContext_Cuda execution_context;
    auto memory_space = native_memory_space( execution_context );
    std::vector<PI8> _u8_host( 2 );
    cudaMemcpyAsync( _u8_host.data(), u8_input_buffer.typed_data(), _u8_host.size() * sizeof( PI8 ), cudaMemcpyDefault, stream );
    cudaStreamSynchronize( stream );
    const PI8 *u8_input = _u8_host.data();
    PI64 *u64_output = u64_output_buffer->typed_data();
    cudaMemsetAsync( u64_output, 0, u64_output_buffer->element_count() * sizeof( PI64 ), stream );
    auto t_di0 = tensor_view_input( CtType<Tuple<TI>>(), memory_space, di0, u8_input[ 1 ] );
    auto t_o0 = tensor_view_output( CtType<Tuple<>>(), memory_space, o0, u8_input[ 0 ] );
    try {
        auto p = Parameters{
            .output = t_o0,
            .input = t_di0,
        };

        using P = DECAYED_TYPE_OF( p ); run_sequential( Range( 1 ), [] HD ( int, P p ) mutable { p.output = p.input[ 1 ]; }, p );
    } catch ( DynamicSizeException de ) {
        PI64 _de_vals[ 2 ] = { 1 + de.num_dynamic_axis, de.needed_size };
        cudaMemcpyAsync( u64_output + 0, _de_vals, 2 * sizeof( PI64 ), cudaMemcpyHostToDevice, stream );
    }
    return xla::ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL( binding_using_P_DECAYED_TYPE_OF_p_run_sequentiaiuW42OoCcWd, impl_using_P_DECAYED_TYPE_OF_p_run_sequentiaiuW42OoCcWd, xla::ffi::Ffi::Bind().Arg<xla::ffi::Buffer<xla::ffi::F64>>().Arg<xla::ffi::Buffer<xla::ffi::U8>>().Ret<xla::ffi::Buffer<xla::ffi::F64>>().Ret<xla::ffi::Buffer<xla::ffi::U64>>().Ctx<xla::ffi::PlatformStream<cudaStream_t>>() );


template<typename T>
nb::capsule EncapsulateFfiCall( T *fn ) {
  static_assert( std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>, "Encapsulated function must be and XLA FFI handler");
  return nb::capsule( reinterpret_cast<void *>( fn ), "xla._CUSTOM_CALL_TARGET" );
}

NB_MODULE( using_P_DECAYED_TYPE_OF_p_run_sequentiaiuW42OoCcWd, m ) {
  m.def( "using_P_DECAYED_TYPE_OF_p_run_sequentiaiuW42OoCcWd", []() { return EncapsulateFfiCall( binding_using_P_DECAYED_TYPE_OF_p_run_sequentiaiuW42OoCcWd ); } );
}
