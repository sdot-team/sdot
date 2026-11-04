<script setup>
// The scaling charts of the performance page ( `CHARTS` of `benchTables.js` ): one log-log panel per row, the threads of the
// CPU machine of `SOURCES.cpu` along x, then the cards of `SOURCES.gpu` as single points at their number of SMs. y: cycles
// ( time x the measured clock ), or the times when a run has no clock. A row with `on: 'gpu'` is a panel of the first card of
// `SOURCES.gpu` that ran it: each series draws there its `card` runs ( `{ m, v }` ), in the series' colour.
import { computed } from 'vue'
import { CHARTS } from './benchTables.js'
import { sources, hwOf, cpuName, gpuName, smsOf, smClockOf, findAll, kindOf } from './benchData.js'

const props = defineProps( { chart: { type: String, required: true } } )

const W = 340, H = 230, L = 50, R = 14, T = 26, B = 34
const SHAPES = [ 'diamond', 'square', 'triangle' ]

// 1.2k, 35k, 1.2M: the ticks and the tooltips
const si = x => {
  for ( const [ d, s ] of [ [ 1e12, 'T' ], [ 1e9, 'G' ], [ 1e6, 'M' ], [ 1e3, 'k' ] ] )
    if ( x >= d ) return `${ +( x / d ).toPrecision( 2 ) }${ s }`
  return `${ +x.toPrecision( 2 ) }`
}
const time = ns => ns >= 1e9 ? `${ +( ns / 1e9 ).toPrecision( 3 ) } s` : ns >= 1e6 ? `${ +( ns / 1e6 ).toPrecision( 3 ) } ms` :
                   ns >= 1e3 ? `${ +( ns / 1e3 ).toPrecision( 3 ) } µs` : `${ +ns.toPrecision( 3 ) } ns`

// ticks of a log axis: 1-2-5 when the range is short, the decades otherwise
function logTicks( lo, hi ) {
  for ( const steps of [ [ 1, 2, 5 ], [ 1, 3 ], [ 1 ] ] ) {
    const t = []
    for ( let e = Math.floor( Math.log10( lo ) ); e <= Math.ceil( Math.log10( hi ) ); e++ )
      for ( const s of steps ) { const v = s * 10 ** e; if ( v >= lo && v <= hi ) t.push( v ) }
    if ( t.length <= 6 ) return t
  }
  return []
}

// a run's clock: the MEDIAN of the clocks its machine recorded at the same thread count ( the CPU ) or on the same card. One
// run's own sample is noisy ( `scaling_cur_freq`, sampled by `benchlib`: 1.5 to 3.4 GHz at the same thread count on an EPYC,
// its serial phases and the sampler's period included ), and the cycles would inherit that noise; `est`: a card with no
// recorded clock, at its usual one
const clockOf = ( mach, kind, r ) => {
  const key = kind === 'gpu' ? 'sm_clock_ghz' : 'clock_ghz'
  const same = mach.rows.filter( q => q.results?.[ key ] > 0 && q.kind === r.kind && ( kind === 'gpu' || q.results.threads === r.results.threads ) )
    .map( q => q.results[ key ] ).sort( ( a, b ) => a - b )
  if ( same.length ) return { clock: same[ same.length >> 1 ] }
  return kind === 'gpu' && smClockOf( hwOf( mach, 'gpu' ) ) ? { clock: smClockOf( hwOf( mach, 'gpu' ) ), est: true } : {}
}

const view = computed( () => {
  const c = CHARTS[ props.chart ]
  if ( !c ) return null
  // ( `x: 'error'`: the relative error of each run along a log axis, read like the points )
  const byErr = c.x === 'error', byN = c.x === 'n' || byErr, byDim = c.x === 'dim'
  const cpu = sources( 'cpu' )[ 0 ], gpus = sources( 'gpu' )
  const cpuThreads = hwOf( cpu, 'cpu' ).physical_cores ?? 'all'
  const cards = gpus.map( ( mach, k ) => { const hw = hwOf( mach, 'gpu' ); return { name: gpuName( hw ), sms: smsOf( hw ), shape: SHAPES[ k ], k } } )
  // a point of a run: x the threads ( or the card's SMs ), or the number of diracs
  // ( over the errors, a curve joins its points in the order of their size, `m`: the error need not decrease with it )
  const point = ( mach, kind, r, ns, card ) => ( { x: byDim ? r.results.dim : byErr ? r.results.error : byN ? r.results.n : card ? card.sms : r.results.threads,
                                                   ord: byErr ? r.results.m ?? 0 : 0, ns, card, ...clockOf( mach, kind, r ) } )
  const curve = ( mach, kind, row, m, v, card ) => {
    const seen = new Set()
    return ( m ? findAll( mach, kind, row, m ) : [] ).map( r => point( mach, kind, r, v( r ), card ) )
      .filter( p => p.x > 0 && p.ns > 0 && !seen.has( p.x ) && seen.add( p.x ) ).sort( ( a, b ) => byErr ? a.ord - b.ord : a.x - b.x )
  }
  const panels = c.rows.map( row => ( {
    // ( a card's panel: named after the first card that ran one of its series )
    label: row.on === 'gpu' ? row.label.replace( '{card}', gpuName( hwOf( gpus.find( g => c.series.some( s => s.card && findAll( g, 'gpu', row, s.card.m ).length ) ) ?? gpus[ 0 ], 'gpu' ) ) )
                            : row.label.replace( '{cpu}', cpuName( hwOf( cpu, 'cpu' ) ) ),
    series: c.series.map( ( s, i ) => {
      const m = typeof s.m === 'function' ? s.m( row ) : s.m
      if ( s.exact ) {
        // an exact method: no error to place it on, a level at its time ( on a card's panel, its `card` run )
        const r = row.on === 'gpu' ? ( s.card ? gpus.map( g => findAll( g, 'gpu', row, s.card.m )[ 0 ] ).find( Boolean ) : null )
                                   : findAll( cpu, 'cpu', row, m )[ 0 ]
        return { label: s.label, i: s.color ?? i, dash: false, line: [], dots: [], cardLines: [], level: r ? s.v( r ) : null }
      }
      if ( row.on === 'gpu' ) {
        // ( the card's runs, read like the CPU's: no marker of a card, the panel says which one it is )
        const runs = mach => s.card ? findAll( mach, 'gpu', row, s.card.m ) : []
        const mach = gpus.find( g => runs( g ).length )
        const line = mach ? curve( mach, 'gpu', row, s.card.m, s.card.v ?? s.v ).map( p => ( { ...p, onCard: gpuName( hwOf( mach, 'gpu' ) ) } ) ) : []
        return { label: s.label, i: s.color ?? i, dash: s.dash, line, dots: [], cardLines: [] }
      }
      const line = curve( cpu, 'cpu', row, m, row.v ?? s.v )
      // a SEQUENTIAL program ( `flat` ): its one run on one thread holds at every thread count, drawn flat and dashed to the
      // last one, with no point where nothing was measured
      const flat = s.flat && !byN && !byDim && line.length === 1 && line[ 0 ].x === 1 && cpuThreads > 1
      if ( flat ) line.push( { ...line[ 0 ], x: cpuThreads, virtual: true } )
      // on the cards: one point each over the threads, a curve each over the diracs
      const onCards = s.gpu ? gpus.map( ( mach, k ) => cards[ k ].sms || byN ? curve( mach, 'gpu', row, s.gpu.m, s.gpu.v, cards[ k ] ) : [] ) : []
      return { label: s.label, i: s.color ?? i, dash: s.dash || flat, line, dots: byN ? [] : onCards.flatMap( l => l.slice( 0, 1 ) ),
               cardLines: byN ? onCards.filter( l => l.length ) : [] }
    } ),
  } ) )
  const pts = s => [ ...s.line, ...s.dots, ...s.cardLines.flat() ]
  const levels = p => p.series.filter( s => s.level > 0 ).map( s => s.level )
  const all = panels.flatMap( p => p.series.flatMap( pts ) )
  const cycles = !byN && !byDim && all.length > 0 && all.every( p => p.clock > 0 )
  for ( const p of all ) p.y = cycles ? p.ns * p.clock : p.ns

  // the same x axis for every panel; y fitted to each
  const xs = [ ...new Set( all.map( p => p.x ) ) ].sort( ( a, b ) => a - b )
  const x0 = byDim ? ( xs[ 0 ] ?? 2 ) - 0.4 : byN ? ( xs[ 0 ] ?? 1e3 ) / 1.6 : 0.8, x1 = byDim ? ( xs.at( -1 ) ?? 6 ) + 0.4 : ( xs.at( -1 ) ?? 24 ) * ( byN ? 1.6 : 1.3 )
  // ( the dimension along a linear axis: semi-log )
  const X = x => L + ( W - L - R ) * ( byDim ? ( x - x0 ) / ( x1 - x0 ) : Math.log( x / x0 ) / Math.log( x1 / x0 ) )
  for ( const p of panels ) {
    const ys = [ ...p.series.flatMap( s => pts( s ).map( q => q.y ) ), ...levels( p ) ]
    if ( !ys.length ) { p.empty = true; continue }
    // the ideal scaling of the first series with a curve ( 1 / threads from its run on one thread ) stays inside the panel
    const first = p.series.map( s => s.line ).find( l => l[ 0 ]?.x === 1 && l.length > 1 )
    if ( !byN && !byDim && first ) {
      p.ideal = { a: first[ 0 ], b: { x: first.at( -1 ).x, y: first[ 0 ].y / first.at( -1 ).x } }
      ys.push( p.ideal.b.y )
    }
    const y0 = Math.min( ...ys ) / 1.6, y1 = Math.max( ...ys ) * 1.6
    const Y = y => T + ( H - T - B ) * ( 1 - Math.log( y / y0 ) / Math.log( y1 / y0 ) )
    p.yticks = logTicks( y0, y1 ).map( v => ( { v, y: Y( v ), label: cycles ? si( v ) : time( v ) } ) )
    const path = l => l.map( ( q, j ) => `${ j ? 'L' : 'M' }${ q.px.toFixed( 1 ) },${ q.py.toFixed( 1 ) }` ).join( '' )
    for ( const s of p.series ) {
      for ( const q of pts( s ) ) {
        q.px = X( q.x ); q.py = Y( q.y )
        const where = byDim ? `${ q.x }D` : byErr ? `${ q.card ? q.card.name : q.onCard ?? 'CPU' }, error ${ q.x.toExponential( 1 ) }`
                    : byN ? `${ q.card ? q.card.name : q.onCard ?? `${ cpuThreads } threads` }, n = ${ nLabel( q.x ) }`
                          : q.card ? `${ q.card.name } ( ${ q.x } SMs )` : `${ q.x } thread${ q.x > 1 ? 's' : '' }`
        q.tip = `${ s.label }, ${ where }: ${ cycles ? `${ si( q.y ) } cycles, ` : '' }${ time( q.ns ) }` +
                `${ q.clock ? ` at ${ q.clock.toFixed( 2 ) } GHz${ q.est ? ' ( estimated: the run recorded none )' : '' }` : '' }`
      }
      s.path = path( s.line )
      if ( s.level > 0 ) {
        s.levelPath = `M${ L },${ Y( s.level ).toFixed( 1 ) }H${ W - R }`
        s.levelTip = `${ s.label }: ${ time( s.level ) }`
      }
      s.cardPaths = s.cardLines.map( l => ( { d: path( l ), card: l[ 0 ].card } ) )
    }
    if ( p.ideal ) p.ideal.path = `M${ X( p.ideal.a.x ) },${ Y( p.ideal.a.y ) }L${ X( p.ideal.b.x ) },${ Y( p.ideal.b.y ) }`
  }
  // the powers of two first ( 128 rather than 96 ), then the other thread counts and the cards' SMs, each label kept where
  // it leaves room ( 64 cores and a 68-SM card: the card's point is named by its marker )
  const pow2 = v => Number.isInteger( Math.log2( v ) ) ? 0 : 1
  const cardXs = new Set( all.filter( p => p.card ).map( p => p.x ) ), kept = [ ]
  // ( over the diracs or the errors: the decades, one in two or three when there are many )
  const decades = () => { const d = []; for ( let e = Math.ceil( Math.log10( x0 ) ); e <= Math.floor( Math.log10( x1 ) ); e++ ) d.push( 10 ** e )
                          const k = Math.ceil( d.length / 6 ); return d.filter( ( _, j ) => j % k === 0 ) }
  for ( const v of byDim ? xs : byN ? decades() : [ ...xs.filter( v => !cardXs.has( v ) ).sort( ( a, b ) => pow2( a ) - pow2( b ) ), ...cardXs ] )
    if ( kept.every( u => Math.abs( X( u ) - X( v ) ) >= 20 ) ) kept.push( v )
  const xticks = kept.map( v => ( { v, x: X( v ), label: byDim ? `${ v }D` : byN ? nLabel( v ) : `${ v }` } ) )
  const usedCards = cards.filter( k => all.some( p => p.card === k ) )
  // OVER THE DIRACS ( or the errors ), one entry per series and machine: a card's curve takes its series' colour and the card's
  // marker ( a colour per card would merge the series on a card )
  const cpuLabel = `${ cpuName( hwOf( cpu, 'cpu' ) ) }, ${ cpuThreads } cores`
  const entries = !byN || byDim ? null : c.series.flatMap( ( s, i ) => {
    const runs = panels.map( p => p.series[ i ] )
    const out = runs.some( r => r.line.length || r.level > 0 ) ? [ { label: c.cores !== false ? `${ s.label }, ${ cpuLabel }` : s.label, i: s.color ?? i, dash: s.dash } ] : []
    for ( const k of usedCards )
      if ( runs.some( r => r.cardLines.some( l => l[ 0 ].card === k ) ) )
        out.push( { label: `${ s.label }, ${ k.name }`, i: s.color ?? i, dash: s.dash, card: k } )
    return out
  } )
  return { panels, xticks, cycles, byN, byDim, byErr, unit: c.unit, cpuName: cpuName( hwOf( cpu, 'cpu' ) ), cpuThreads, cores: c.cores !== false, xLabel: c.xLabel, total: c.total,
           series: c.series.map( ( s, i ) => ( { label: s.label, i: s.color ?? i, dash: s.dash } ) ), cards: usedCards, entries,
           hasIdeal: panels.some( p => p.ideal ), estimated: all.some( p => p.est ) }
} )

const nLabel = n => n.toExponential( 0 ).replace( 'e+', 'e' )

// the marker of a card's point, centred on ( x, y )
const shape = ( kind, x, y, r = 5.5 ) => kind === 'square' ? `M${ x - r * .8 },${ y - r * .8 }h${ r * 1.6 }v${ r * 1.6 }h${ -r * 1.6 }Z`
  : kind === 'triangle' ? `M${ x },${ y - r }L${ x + r },${ y + r * .8 }L${ x - r },${ y + r * .8 }Z`
  : `M${ x },${ y - r }L${ x + r },${ y }L${ x },${ y + r }L${ x - r },${ y }Z`
</script>

<template>
  <div v-if="view" class="bench-chart">
    <div v-if="view.entries" class="legend">
      <span v-for="e in view.entries" :key="e.label"><svg width="26" height="14"><path d="M1,7H25" :class="[ `c${ e.i }`, { dash: e.dash } ]" class="line" />
        <path v-if="e.card" :d="shape( e.card.shape, 13, 7, 5.5 )" :class="[ `c${ e.i }`, { hollow: e.dash } ]" class="dot gpu" /><circle v-else cx="13" cy="7" r="4" :class="`c${ e.i }`" class="dot" /></svg>{{ e.label }}</span>
    </div>
    <div v-else class="legend">
      <span v-for="s in view.series" :key="s.label"><svg width="26" height="12"><path d="M1,6H25" :class="[ `c${ s.i }`, { dash: s.dash } ]" class="line" /><circle cx="13" cy="6" r="4" :class="`c${ s.i }`" class="dot" /></svg>{{ s.label }}</span>
      <span v-if="view.byN && !view.byDim && view.cores"><svg width="14" height="14"><circle cx="7" cy="7" r="4" class="cpu-mark" /></svg>{{ view.cpuName }}, {{ view.cpuThreads }} cores</span>
      <span v-for="k in view.cards" :key="k.name"><svg width="16" height="16"><path :d="shape( k.shape, 8, 8, 6 )" :class="`g${ k.k }`" class="card" /></svg>sdot, {{ k.name }}<template v-if="!view.byN"> ( {{ k.sms }} SMs )</template></span>
      <span v-if="view.hasIdeal"><svg width="26" height="12"><path d="M1,6H25" class="ideal" /></svg>ideal scaling</span>
    </div>
    <div class="panels">
      <figure v-for="p in view.panels" :key="p.label">
        <svg :viewBox="`0 0 ${ W } ${ H }`" role="img" :aria-label="p.label">
          <text :x="L" y="15" class="title">{{ p.label }}</text>
          <template v-if="!p.empty">
            <g class="grid">
              <line v-for="t in p.yticks" :key="t.v" :x1="L" :x2="W - R" :y1="t.y" :y2="t.y" />
            </g>
            <text v-for="t in p.yticks" :key="'y' + t.v" :x="L - 6" :y="t.y + 4" class="tick" text-anchor="end">{{ t.label }}</text>
            <text v-for="t in view.xticks" :key="'x' + t.v" :x="t.x" :y="H - B + 16" class="tick" text-anchor="middle">{{ t.label }}</text>
            <line :x1="L" :x2="W - R" :y1="H - B" :y2="H - B" class="axis" />
            <text :x="( L + W - R ) / 2" :y="H - 4" class="axis-label" text-anchor="middle">{{ view.byDim ? 'dimension' : view.byN ? ( view.xLabel ?? 'diracs' ) : 'threads · SMs' }}</text>
            <path v-if="p.ideal" :d="p.ideal.path" class="ideal" />
            <g v-for="s in p.series" :key="s.label">
              <path :d="s.path" :class="[ `c${ s.i }`, { dash: s.dash } ]" class="line" />
              <path v-if="s.levelPath" :d="s.levelPath" :class="`c${ s.i }`" class="line level"><title>{{ s.levelTip }}</title></path>
              <path v-for="l in s.cardPaths" :key="l.card.name" :d="l.d" :class="[ `c${ s.i }`, { dash: s.dash } ]" class="line gpu" />
              <g v-for="l in s.cardLines" :key="'l' + l[ 0 ].card.name"><g v-for="q in l" :key="q.x"><path :d="shape( q.card.shape, q.px, q.py, 6.5 )" :class="[ `c${ s.i }`, { hollow: s.dash } ]" class="dot gpu" /><circle :cx="q.px" :cy="q.py" r="10" class="hit"><title>{{ q.tip }}</title></circle></g></g>
              <g v-for="q in s.line.filter( q => !q.virtual )" :key="q.x"><circle :cx="q.px" :cy="q.py" r="4" :class="`c${ s.i }`" class="dot" /><circle :cx="q.px" :cy="q.py" r="10" class="hit"><title>{{ q.tip }}</title></circle></g>
              <g v-for="q in s.dots" :key="'g' + q.x"><path :d="shape( q.card.shape, q.px, q.py, 6.5 )" :class="[ `g${ q.card.k }`, { hollow: s.dash } ]" class="dot gpu" /><circle :cx="q.px" :cy="q.py" r="10" class="hit"><title>{{ q.tip }}</title></circle></g>
            </g>
          </template>
          <text v-else :x="W / 2" :y="H / 2" class="tick" text-anchor="middle">no run</text>
        </svg>
      </figure>
    </div>
    <p class="bench-note">{{ view.byDim ? `Time, ${ view.unit }: semi-log, the dimension along a linear axis.` : view.byErr ? `Time against the relative error, ${ view.unit }; a curve joins the runs of a method in the order of their size.`
                           : view.byN ? ( view.total ? `Total time ${ view.unit }: a slope of one decade per decade is linear.` : `Time ${ view.unit }: flat is linear in the number of diracs.` )
                           : view.cycles ? `Cycles ${ view.unit }: the time times the clock, the median of the clocks measured on the machine at that thread count ( the cores' on the CPU, the SMs' on a card${ view.estimated ? '; a card that recorded none at its usual clock' : '' } ): one run's sample is too noisy.`
                           : `Time ${ view.unit }: these runs recorded no clock, so the cycles cannot be computed.` }}
      {{ view.byDim ? 'Hover' : 'Log-log; hover' }} a point for its time<template v-if="view.cycles"> and clock</template>.</p>
  </div>
  <p v-else><em>unknown chart {{ chart }}</em></p>
</template>

<style scoped>
.bench-chart {
  --s0: #2a78d6; --s1: #eb6834; --s2: #1baf7a; --s3: #eda100; --s4: #e87ba4;
  /* the cards over the threads: a colour each, apart from every CPU series ( and a marker each: `SOURCES.gpu` order ); over n, their series' colour */
  --g0: #6a2fb0; --g1: #c1297e; --g2: #336314;
  --ring: var( --vp-c-bg ); --ink: var( --vp-c-text-1 ); --ink2: var( --vp-c-text-2 ); --rule: var( --vp-c-divider );
  margin: 1em 0;
}
:global( html.dark ) .bench-chart { --s0: #3987e5; --s1: #d95926; --s2: #199e70; --s3: #c98500; --s4: #d55181; --g0: #8654bc; --g1: #dd51ad; --g2: #4f9a2a; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 18px; font-size: 0.85em; color: var( --ink2 ); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.panels { display: grid; grid-template-columns: repeat( auto-fill, minmax( 280px, 1fr ) ); gap: 8px 16px; margin-top: 8px; }
figure { margin: 0; }
figure svg { display: block; width: 100%; height: auto; overflow: visible; }
.legend svg { display: inline-block; flex: none; }
.title { font-size: 12.5px; font-weight: 600; fill: var( --ink ); }
.tick, .axis-label { font-size: 10.5px; fill: var( --ink2 ); font-variant-numeric: tabular-nums; }
.grid line { stroke: var( --rule ); stroke-width: 1; }
.axis { stroke: var( --ink2 ); stroke-width: 1; }
.line { fill: none; stroke-width: 2; stroke-linejoin: round; }
.dot { stroke: var( --ring ); stroke-width: 2; }
.card.g0, .dot.gpu.g0 { fill: var( --g0 ); } .card.g1, .dot.gpu.g1 { fill: var( --g1 ); } .card.g2, .dot.gpu.g2 { fill: var( --g2 ); }
.cpu-mark { fill: var( --ink2 ); }
.ideal { fill: none; stroke: var( --ink2 ); stroke-width: 1.2; stroke-dasharray: 4 4; opacity: 0.7; }
.hit { fill: transparent; }
.line.dash { stroke-dasharray: 6 4; }
.line.level { stroke-dasharray: 2 3; stroke-width: 2.5; }
.line.c0 { stroke: var( --s0 ); } .dot.c0 { fill: var( --s0 ); }
.line.c1 { stroke: var( --s1 ); } .dot.c1 { fill: var( --s1 ); }
.line.c2 { stroke: var( --s2 ); } .dot.c2 { fill: var( --s2 ); }
.line.c3 { stroke: var( --s3 ); } .dot.c3 { fill: var( --s3 ); }
.line.c4 { stroke: var( --s4 ); } .dot.c4 { fill: var( --s4 ); }
/* a prepared point on a card ( a dashed series ): the marker hollow, the from-scratch one filled */
.dot.gpu.hollow { fill: var( --ring ); stroke-width: 2; } .dot.gpu.hollow.c0 { stroke: var( --s0 ); } .dot.gpu.hollow.c1 { stroke: var( --s1 ); } .dot.gpu.hollow.c2 { stroke: var( --s2 ); } .dot.gpu.hollow.c3 { stroke: var( --s3 ); } .dot.gpu.hollow.c4 { stroke: var( --s4 ); } .dot.gpu.hollow.g0 { stroke: var( --g0 ); } .dot.gpu.hollow.g1 { stroke: var( --g1 ); } .dot.gpu.hollow.g2 { stroke: var( --g2 ); }
.line.gpu { stroke-width: 2.5; } .line.gpu.g0 { stroke: var( --g0 ); } .line.gpu.g1 { stroke: var( --g1 ); } .line.gpu.g2 { stroke: var( --g2 ); }
.bench-note { font-size: 0.85em; color: var( --vp-c-text-2 ); margin-top: 0.3em; }
</style>
