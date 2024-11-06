#include <tl/support/ASSERT.h>
#include "Mpi.h"

namespace sdot {

//
class NoMpi : public Mpi {
public:
    virtual int   rank    () const override { return 0; }
    virtual int   size    () const override { return 1; }

protected:
    virtual void  _scatter( PI src_rank, Span<B> value ) override {}
    virtual void  _gather ( PI tgt_rank, Span<B> outputs, CstSpan<B> input ) override { ASSERT( outputs.size() == input.size() ); std::memcpy( outputs.data(), input.data(), input.size() ); }
};

static NoMpi inst_mpi_abstraction;
Mpi *mpi = &inst_mpi_abstraction;

} // namespace sdot
