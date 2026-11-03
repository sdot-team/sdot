<script setup>
// One table of the performance page ( `benchTables.js` ), filled from the files of `bench/page.py`. `table="setup"`: the
// hardware itself. The CPU columns come from the machine of `SOURCES.cpu`, the GPU ones from the cards of `SOURCES.gpu`, side
// by side ( one column per card, headed by its name ), then merged: one column per program and kind, `a / b / c`.
import { computed } from 'vue'
import { TABLES } from './benchTables.js'
import { sources, envOf, hwOf, cpuName, gpuName, find } from './benchData.js'

const props = defineProps( { table: { type: String, required: true } } )

const setup = computed( () => [ 'cpu', 'gpu' ].flatMap( kind => sources( kind ).map( mach => {
  const hw = hwOf( mach, kind ), e = envOf( mach, kind )
  return { kind, name: kind === 'cpu' ? cpuName( hw ) : gpuName( hw ), cores: hw.physical_cores, ram: hw.ram_gb,
           host: cpuName( hw ), kernels: hw.kernels, date: e?.date, commit: e?.commit }
} ) ) )

const grid = computed( () => {
  const t = TABLES[ props.table ]
  if ( !t ) return null
  const cpu = sources( 'cpu' )[ 0 ]
  // each column once for the CPU machine, or once per card
  const cols = []
  for ( const c of t.cols ) {
    const src = c.src ?? c.kind
    for ( const mach of src === 'cpu' ? [ cpu ].filter( Boolean ) : sources( 'gpu' ) ) {
      const get = ( kind, m, row ) => find( kind === 'cpu' ? cpu : mach, kind, row, m )
      const cell = row => {
        if ( c.calc ) return { text: c.calc( ( kind, m ) => get( kind, m, row ), row ) }
        const run = get( c.kind, c.m, row )
        if ( !run ) return { text: null }
        const raw = ( c.v ?? row.v )( run )
        return { text: ( c.f ?? row.f )( raw ), key: typeof c.best === 'function' ? c.best( raw ) : c.best ? raw : null }
      }
      const cells = t.rows.map( cell )
      if ( cells.every( x => x.text == null ) ) continue
      const first = t.rows.map( row => c.kind && get( c.kind, c.m, row ) ).find( Boolean )
      const label = src === 'gpu' ? gpuName( hwOf( mach, 'gpu' ) ) : typeof c.label === 'function' ? c.label( first ) : c.label
      cols.push( { group: c.group ?? '', src, label, cells, merge: c.merge ?? ( src === 'gpu' ? `gpu ${ t.cols.indexOf( c ) }` : null ) } )
    }
  }
  // the shaded cells: per row and per kind, the smallest of the `best` columns
  t.rows.forEach( ( _, j ) => {
    for ( const src of [ 'cpu', 'gpu' ] ) {
      const cand = cols.filter( c => c.src === src && typeof c.cells[ j ].key === 'number' )
      if ( cand.length < 2 ) continue
      const min = Math.min( ...cand.map( c => c.cells[ j ].key ) )
      cand.filter( c => c.cells[ j ].key === min ).forEach( c => c.cells[ j ].best = true )
    }
  } )
  // the columns shown: consecutive ones of the same `merge` key in one, their values side by side
  const shown = []
  for ( const c of cols ) {
    const last = shown.at( -1 )
    if ( c.merge && last?.merge === c.merge && last.group === c.group ) last.parts.push( c )
    else shown.push( { group: c.group, merge: c.merge, parts: [ c ] } )
  }
  for ( const s of shown ) {
    const labels = s.parts.map( c => c.label ).join( ' / ' )
    s.label = s.merge === 'threads' ? `${ labels } thread${ labels === '1' ? '' : 's' }` : labels
    s.cells = t.rows.map( ( _, j ) => s.parts.map( c => c.cells[ j ] ) )
  }
  const groups = []
  for ( const c of shown ) {
    if ( groups.length && groups.at( -1 ).head === c.group ) groups.at( -1 ).span++
    else groups.push( { head: c.group, span: 1 } )
  }
  return { note: t.note, rows: t.rows, cols: shown, groups: groups.some( g => g.head ) ? groups : null }
} )
</script>

<template>
  <div v-if="table === 'setup'">
    <table v-if="setup.length">
      <thead><tr><th></th><th>hardware</th><th>cores</th><th>RAM</th><th>host CPU</th><th>kernels</th><th>measured</th><th>commit</th></tr></thead>
      <tbody>
        <tr v-for="( s, i ) in setup" :key="i">
          <td>{{ s.kind.toUpperCase() }}</td><td>{{ s.name }}</td><td>{{ s.kind === 'cpu' ? s.cores : '' }}</td>
          <td>{{ s.kind === 'cpu' && s.ram ? Math.round( s.ram ) + ' GB' : '' }}</td><td>{{ s.kind === 'gpu' ? s.host : '' }}</td>
          <td>{{ s.kernels }}</td><td>{{ s.date }}</td><td><code v-if="s.commit">{{ s.commit }}</code></td>
        </tr>
      </tbody>
    </table>
    <p v-else><em>No results yet: run <code>bench/page.py</code>.</em></p>
  </div>
  <div v-else-if="grid">
    <table v-if="grid.cols.length">
      <thead>
        <tr v-if="grid.groups"><th></th><th v-for="( g, i ) in grid.groups" :key="i" :colspan="g.span" class="group">{{ g.head }}</th></tr>
        <tr><th></th><th v-for="( c, i ) in grid.cols" :key="i">{{ c.label }}</th></tr>
      </thead>
      <tbody>
        <tr v-for="( row, j ) in grid.rows" :key="row.label">
          <td>{{ row.label }}</td>
          <td v-for="( c, i ) in grid.cols" :key="i" class="num"><template v-for="( p, k ) in c.cells[ j ]" :key="k"><span v-if="k" class="sep"> / </span><span :class="{ best: p.best }">{{ p.text ?? '–' }}</span></template></td>
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
span.best { background: var( --vp-c-brand-soft ); border-radius: 3px; padding: 0 3px; }
span.sep { color: var( --vp-c-text-3 ); }
th.group { text-align: center; }
.bench-note { font-size: 0.85em; color: var( --vp-c-text-2 ); margin-top: -0.5em; }
</style>
