// The runs of the result files of `bench/page.py`, as the tables ( `BenchTable.vue` ) and the charts ( `BenchChart.vue` )
// read them: the machines of `SOURCES`, what each ran on, and the run matching a row and a column ( or a series ).
import { data as files } from '../../guide/bench.data.js'
import { SOURCES } from './benchTables.js'

const byName = Object.fromEntries( files.map( m => [ m.name, m ] ) )
export const sources = kind => SOURCES[ kind ].map( n => byName[ n ] ).filter( Boolean )

// a run's kind: its own ( an environment with a card runs both ), else its environment's ( the files before that )
export const kindOf = ( mach, run ) => run.kind ?? mach.envs?.[ run.env ]?.kind ?? 'cpu'
// the kinds an environment ran, each with what it ran on
const kindsOf = e => e.kinds ?? { [ e.kind ?? 'cpu' ]: { hw: e.hw } }
// what `mach` ran its runs of `kind` on, and when
export const envOf = ( mach, kind ) => Object.values( mach.envs ?? {} ).find( e => kindsOf( e )[ kind ] )
export const hwOf = ( mach, kind ) => kindsOf( envOf( mach, kind ) ?? {} )[ kind ]?.hw ?? {}

// the names on the page: the processor or the card, without the vendor's marks
export const cpuName = hw => hw.cpu?.replace( /\((R|TM)\)/g, '' ).replace( /^Intel |^AMD /, '' ).replace( / CPU @.*| @.*/, '' ).replace( / \d+-Core Processor$/, '' ).trim()
export const gpuName = hw => hw.gpus?.split( ',' )[ 0 ].replace( /^NVIDIA /, '' ).replace( /^GeForce /, '' ).replace( / \d+ ?GB.*$/, '' ).trim()

// the SMs of a card: recorded since the clocks are ( `hw_sms` ), from its name before
const SMS = { 'RTX 2080 Ti': 68, 'H100': 132, 'H100 PCIe': 114, 'A100-SXM4': 108, 'RTX A6000': 84 }
export const smsOf = hw => hw.sms ?? SMS[ gpuName( hw ) ]
// the SM clock of a card under load, for its runs that recorded none ( measured with `nvidia-smi` during a solve )
const SM_CLOCKS = { 'RTX 2080 Ti': 1.83, 'H100': 1.98 }
export const smClockOf = hw => SM_CLOCKS[ gpuName( hw ) ]

// `threads`: a count, or `cores` ( all the physical cores: -1, or their number )
const threadsWanted = ( mach, want ) => want === 'cores' ? [ -1, hwOf( mach, 'cpu' ).physical_cores ] : want

function matches( mach, run, m ) {
  for ( let [ k, want ] of Object.entries( m ) ) {
    if ( want === undefined ) continue                    // ( a key of the row that the series lets go )
    const have = k === 'job' ? run.job : run.params?.[ k ]
    if ( k === 'threads' ) want = threadsWanted( mach, want )
    if ( Array.isArray( want ) ? !want.includes( have ) : have !== want ) return false
  }
  return true
}

// the runs of `mach` in an environment of `kind` matching the row ( with its per-kind override ) and `m`
export function findAll( mach, kind, row, m ) {
  const over = row[ kind ]
  if ( over === null || !mach ) return []
  const all = { ...row.m, ...( over ?? {} ), ...m }
  return mach.rows.filter( r => r.status === 'PASS' && kindOf( mach, r ) === kind && matches( mach, r, all ) )
}
export const find = ( mach, kind, row, m ) => findAll( mach, kind, row, m )[ 0 ]
