import { galleryCards, SECTIONS } from '../.vitepress/gallery.js'
import tagWeights from './tag-weights.json'

// the cards of the gallery, by section ( see .vitepress/gallery.js ), and the tags of all of them
export default {
  watch: [ './*.md', '../guide/**/*.md' ],
  load() {
    const cards = galleryCards()
    const sections = SECTIONS.map( s => ( { ...s, cards: cards[ s.key ] } ) )
    const all = sections.flatMap( s => s.cards )

    // Count how many cards carry each tag
    const freq = {}
    all.forEach( e => e.tags.forEach( t => { freq[ t ] = ( freq[ t ] ?? 0 ) + 1 } ) )

    // Collect unique tags and sort:
    //   1. explicit weight from tag-weights.json (ascending)
    //   2. unweighted tags (weight = Infinity) sorted by frequency desc, then alpha
    const tags = [ ...new Set( all.flatMap( e => e.tags ) ) ]
      .sort( ( a, b ) => {
        const wa = tagWeights[ a ] ?? Infinity
        const wb = tagWeights[ b ] ?? Infinity
        if ( wa !== wb ) return wa - wb
        const diff = ( freq[ b ] ?? 0 ) - ( freq[ a ] ?? 0 )
        return diff !== 0 ? diff : a.localeCompare( b )
      } )

    return { sections, tags }
  }
}
