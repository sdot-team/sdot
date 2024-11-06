#include "Mpi.h"

namespace sdot {

//
class NoMpi : public Mpi {
public:
    virtual int           rank                   () const override { return 0; }
    virtual int           size                   () const override { return 1; }

protected:
    virtual void          _broadcast             ( Span<B> value, PI src_rank = 0 ) override {}
    virtual void          _gather                ( Span<B> output, CstSpan<B> input, PI tgt_rank = 0 ) override {}
};

static NoMpi inst_mpi_abstraction;
Mpi *mpi = &inst_mpi_abstraction;

} // namespace sdot
