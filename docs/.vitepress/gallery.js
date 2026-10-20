// The cards of the examples gallery, read from the frontmatter of the pages -- the one source of both the gallery
// ( `examples/examples.data.js` ) and the 'Examples' sidebar ( `config.mts` ).
//
//   * `examples/*.md` are the use cases ( self-contained applications );
//   * a guide page with `gallery: feature` is a feature, shown at work by the examples of its page.
//
// Frontmatter: `title` ( default: the first `# ` heading ), `description`, `tags: [ a, b ]`, `image` ( a path under `public/`,
// default `/examples/img/<slug>.png` for an example ), `order`, `draft: true` to hide a page.
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const DOCS = path.resolve( path.dirname( fileURLToPath( import.meta.url ) ), '..' )

export const GALLERY_DIRS = [ path.join( DOCS, 'examples' ), path.join( DOCS, 'guide' ) ]

export const SECTIONS = [
  { key: 'feature' , text: 'Features' , blurb: 'One feature per page of the guide, shown at work.' },
  { key: 'use-case', text: 'Use cases', blurb: 'Self-contained, runnable applications.' },
]

// flat `key: value` lines; `[ a, b ]` is a list, quotes are removed
function frontmatter( src ) {
  const fm = src.match( /^---\r?\n([\s\S]*?)\r?\n---/ )?.[ 1 ] ?? ''
  const res = {}
  for ( const line of fm.split( /\r?\n/ ) ) {
    const m = line.match( /^([\w-]+):\s*(.*?)\s*$/ )
    if ( !m ) continue
    let v = m[ 2 ].replace( /^(['"])(.*)\1$/, '$2' )
    if ( v.startsWith( '[' ) && v.endsWith( ']' ) )
      v = v.slice( 1, -1 ).split( ',' ).map( t => t.trim() ).filter( t => t )
    res[ m[ 1 ] ] = v
  }
  return res
}

function mdFiles( dir ) {
  return fs.readdirSync( dir, { withFileTypes: true } ).flatMap( e =>
    e.isDirectory() ? mdFiles( path.join( dir, e.name ) ) : e.name.endsWith( '.md' ) ? [ path.join( dir, e.name ) ] : [] )
}

function card( file, kind ) {
  const src  = fs.readFileSync( file, 'utf8' )
  const fm   = frontmatter( src )
  const rel  = path.relative( DOCS, file ).slice( 0, -3 ).split( path.sep ).join( '/' )
  const slug = rel.split( '/' ).pop()
  return {
    kind,
    title:       fm.title ?? src.match( /^# (.*)$/m )?.[ 1 ].trim() ?? slug,
    description: fm.description ?? '',
    tags:        fm.tags ?? [],
    order:       Number( fm.order ?? 999 ),
    draft:       fm.draft === 'true',
    link:        '/' + rel,
    image:       fm.image ?? ( kind === 'use-case' ? `/examples/img/${ slug }.png` : null ),
    gallery:     fm.gallery,
  }
}

// { 'feature': [ cards ], 'use-case': [ cards ] }, each sorted by `order` then title
export function galleryCards() {
  const useCases = fs.readdirSync( GALLERY_DIRS[ 0 ] )
    .filter( f => f.endsWith( '.md' ) && f !== 'index.md' )
    .map( f => card( path.join( GALLERY_DIRS[ 0 ], f ), 'use-case' ) )
  const features = mdFiles( GALLERY_DIRS[ 1 ] )
    .map( f => card( f, 'feature' ) )
    .filter( c => c.gallery === 'feature' )
  const sorted = cards => cards
    .filter( c => !c.draft )
    .sort( ( a, b ) => a.order - b.order || a.title.localeCompare( b.title ) )
  return { 'feature': sorted( features ), 'use-case': sorted( useCases ) }
}

// the 'Examples' sidebar: the gallery, then one group per section
export function gallerySidebar() {
  const cards = galleryCards()
  return [
    { text: 'Examples', items: [ { text: 'Gallery', link: '/examples/' } ] },
    ...SECTIONS.map( s => ( {
      text:  s.text,
      items: cards[ s.key ].map( c => ( { text: c.title, link: c.link } ) ),
    } ) ),
  ]
}
