<script setup>
// One table of the performance page ( `benchTables.js` ), filled from the files of `bench/page.py`. `table="setup"`: the
// machines themselves.
import { computed } from 'vue'
import { data as machines } from '../../guide/bench.data.js'
import { TABLES } from './benchTables.js'

const props = defineProps( { table: { type: String, required: true } } )

// a run's kind: its own ( an environment with a card runs both ), else its environment's ( the files before that )
const kindOf = ( mach, run ) => run.kind ?? mach.envs?.[ run.env ]?.kind ?? 'cpu'
// the kinds an environment ran, each with what it ran on
const kindsOf = e => e.kinds ?? { [ e.kind ?? 'cpu' ]: { hw: e.hw } }

function matches( run, m ) {
  for ( const [ k, want ] of Object.entries( m ) ) {
    const have = k === 'job' ? run.job : run.params?.[ k ]
    if ( Array.isArray( want ) ? !want.includes( have ) : have !== want ) return false
  }
  return true
}

// the run of `mach` in an environment of `kind` matching the row ( with its per-kind override ) and `m`
function find( mach, kind, row, m ) {
  const over = row[ kind ]
  if ( over === null ) return undefined
  const all = { ...row.m, ...( over ?? {} ), ...m }
  return mach.rows.find( r => r.status === 'PASS' && kindOf( mach, r ) === kind && matches( r, all ) )
}

const setup = computed( () => machines.flatMap( mach => Object.entries( mach.envs ?? {} ).flatMap( ( [ env, e ] ) =>
  Object.entries( kindsOf( e ) ).map( ( [ kind, { hw } ] ) => ( {
    name: mach.name, env, kind, cpu: hw?.cpu, cores: hw?.physical_cores ?? hw?.cores, kernels: hw?.kernels, ram: hw?.ram_gb, gpus: hw?.gpus, date: e.date, commit: e.commit,
  } ) ) ) ) )

const grid = computed( () => {
  const t = TABLES[ props.table ]
  if ( !t ) return null
  // the columns, repeated for every machine that has data for them
  const cols = []
  for ( const mach of machines )
    for ( const c of t.cols ) {
      const cell = row => {
        if ( c.calc ) return c.calc( ( kind, m ) => find( mach, kind, row, m ) )
        const run = find( mach, c.kind, row, c.m )
        return run ? c.f( c.v( run ) ) : null
      }
      const values = t.rows.map( cell )
      if ( values.every( v => v == null ) ) continue
      const first = t.rows.map( row => c.kind && find( mach, c.kind, row, c.m ) ).find( Boolean )
      cols.push( { machine: mach.name, label: typeof c.label === 'function' ? c.label( first ) : c.label, values } )
    }
  const groups = []
  for ( const c of cols ) {
    if ( groups.length && groups.at( -1 ).machine === c.machine ) groups.at( -1 ).span++
    else groups.push( { machine: c.machine, span: 1 } )
  }
  return { note: t.note, rows: t.rows, cols, groups }
} )
</script>

<template>
  <div v-if="table === 'setup'">
    <table v-if="setup.length">
      <thead><tr><th>machine</th><th>environment</th><th>CPU</th><th>cores</th><th>RAM</th><th>GPU</th><th>CPU kernels</th><th>measured</th><th>commit</th></tr></thead>
      <tbody>
        <tr v-for="s in setup" :key="s.name + s.env + s.kind">
          <td>{{ s.name }}</td><td><code>{{ s.env }}</code> ( {{ s.kind }} )</td><td>{{ s.cpu }}</td><td>{{ s.cores }}</td>
          <td>{{ s.ram ? s.ram + ' GB' : '' }}</td><td>{{ s.kind === 'gpu' ? s.gpus : '' }}</td><td>{{ s.kind === 'cpu' ? s.kernels : '' }}</td><td>{{ s.date }}</td><td><code>{{ s.commit }}</code></td>
        </tr>
      </tbody>
    </table>
    <p v-else><em>No results yet: run <code>bench/page.py</code>.</em></p>
  </div>
  <div v-else-if="grid">
    <table v-if="grid.cols.length">
      <thead>
        <tr v-if="machines.length > 1"><th></th><th v-for="g in grid.groups" :key="g.machine" :colspan="g.span">{{ g.machine }}</th></tr>
        <tr><th></th><th v-for="( c, i ) in grid.cols" :key="i">{{ c.label }}</th></tr>
      </thead>
      <tbody>
        <tr v-for="( row, j ) in grid.rows" :key="row.label">
          <td>{{ row.label }}</td>
          <td v-for="( c, i ) in grid.cols" :key="i" class="num">{{ c.values[ j ] ?? '–' }}</td>
        </tr>
      </tbody>
    </table>
    <p v-else><em>No results yet for this table: run <code>bench/page.py</code>.</em></p>
    <p class="bench-note">{{ grid.note }}</p>
  </div>
  <p v-else><em>unknown table {{ table }}</em></p>
</template>

<style scoped>
td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.bench-note { font-size: 0.85em; color: var( --vp-c-text-2 ); margin-top: -0.5em; }
</style>
