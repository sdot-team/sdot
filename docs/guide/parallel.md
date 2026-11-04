# Running in Parallel

A power diagram can be computed by several devices or processes at once. The call is still `PowerDiagram( ... )`. What
decides is the **positions**: split along the seeds between several *ranks*, they give one box of seeds per rank.

::: code-group

<<< @/snippets/parallel_jax.py [JAX ( devices )]

<<< @/snippets/parallel_torch.py [PyTorch ( processes )]

<<< @/snippets/parallel_mpi.py [MPI ( processes )]

<<< @/snippets/parallel_boxes.py [One device ( boxes in turn )]

:::

| Context | The ranks | How the positions are split | Launched with | `measures` |
|---|---|---|---|---|
| JAX | the devices of a `Mesh` ( `loom.JaxRanks( mesh )` ) | a JAX array with a `NamedSharding` along the seeds | `python script.py` | one JAX array, sharded like the positions |
| PyTorch | the processes of `torch.distributed` ( `loom.TorchRanks()` ) | `loom.spread( block, ranks )` in each process | `torchrun --nproc-per-node 4 script.py` | this process's block |
| MPI ( NumPy, CuPy ) | the processes of a communicator ( `loom.MpiRanks( comm )` ) | `loom.spread( block, ranks )` in each process | `mpiexec -n 4 python script.py` | this process's block |
| One device | — | `nb_boxes = K` | `python script.py` | an array |

## What is spread

- **JAX** is one process driving every device. A JAX array already sharded along its first axis ( `NamedSharding( mesh, P( "p",
  None ) )` ) counts as spread as it is. `loom.spread( array, loom.JaxRanks( mesh ) )` does the same from a host array.
- **PyTorch and MPI** run one process per rank. Each process holds **its own block** of the seeds and says so with `loom.spread(
  block, ranks )`. `loom.spread_host( whole_array, ranks )` is the other way round: every process passes the same whole array and
  keeps its block.
- The result is spread like the positions. With PyTorch and MPI it is a loom tensor whose `.value` is the block of the process,
  and whose reductions are over all the processes: `m.sum()`, `m.max()`, `m.mean()`... give every process the same result.
- `loom.gather_to_host( m )` gives the whole of a spread result on the host.

With PyTorch and MPI, a reduction and `gather_to_host` are collectives: every process must call them.

The weights follow the positions: spread the same way, or given whole.

## How it works

Each rank builds the cells of its own seeds, starting from the whole domain. A cell that its seeds can no longer cut, but that the
seeds of another rank still could, **travels** there: its vertices go in a message, and the cell goes on being cut on the other side.
A cell only shrinks, so it never goes back to a rank it has visited. When no cell is travelling any more, the measure of each cell
goes back to the rank of its seed.

No seed is copied from one rank to another: what crosses the boundaries are cells. Two sizes are not known in advance: the vertices
of a travelling cell ( `msg_capacity`, 16 by default ) and the cells a rank sends to another in one round ( `inbox_capacity`, 64 by
default ). When one of them is too small, the computation runs again with twice as much, and the diagram keeps the new size for its
next calls.

## The same code for every framework

sdot writes the distributed computation **once**, as a *program per rank*, and [loom](https://hleclerc.github.io/loom/) runs it in
each context: in a `shard_map` under `jit` for JAX, as it is in each process for PyTorch and MPI. A program reads the blocks of its
rank and talks to the others through a few operations:

- `ranks.rank()`, `ranks.size`, `ranks.all_sum( x )`, `ranks.exchange( buf )` ( `buf[ j ]` goes to rank `j` );
- `loom.route( dest, rows, size, capacity )` packs rows by destination for an exchange, and `loom.scatter_add( target, idx, vals )`
  adds values at indices. Both are loom kernels, the same code everywhere;
- `loom.while_loop( cond, body, carry )`, and `loom.Capacity` for the sizes found too small along the way.

<<< @/snippets/parallel_program.py

## For now

- 2D, with a domain ( `boundaries` ), and `measures` only: no gradient yet, and the other arguments of `PowerDiagram` ( a
  `distribution`, a periodicity... ) are refused.
- The number of seeds must split evenly between the ranks, and the ranks should be a power of two.
- The boxes are still made on the host, from the whole cloud ( gathered from the processes when they each hold a block ).
- At most 52 ranks or boxes.

The steps that remain are on the [roadmap](./roadmap).
