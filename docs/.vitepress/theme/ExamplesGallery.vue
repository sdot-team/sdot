<script setup>
import { ref, computed } from 'vue'
import { withBase } from 'vitepress'
import { data } from '../../examples/examples.data.js'

const allSections = data.sections   // features, then use cases ( see .vitepress/gallery.js )
const allTags     = data.tags       // pre-sorted by loader

const PLACEHOLDER = withBase( '/examples/img/patate.png' )
function imgSrc( path ) { return path ? withBase( path ) : PLACEHOLDER }

const activeTags = ref( [] )

function toggleTag( tag ) {
  activeTags.value = activeTags.value.includes( tag )
    ? activeTags.value.filter( t => t !== tag )
    : [ ...activeTags.value, tag ]
}

function clearTags() { activeTags.value = [] }
function isActive( tag ) { return activeTags.value.includes( tag ) }

// one flat list: the first card of each section carries the section, labelled above it in the grid
const cards = computed( () => allSections.flatMap( s => s.cards
  .filter( e => activeTags.value.every( t => e.tags.includes( t ) ) )
  .map( ( e, i ) => ( { ...e, section: i === 0 ? s : null } ) )
) )
</script>

<template>
  <div class="gallery-page">
    <header class="gallery-header">
      <h1>Examples</h1>
      <p>The features of SDOT, each shown at work on a page of the guide, and use cases that put them together.</p>
    </header>

    <!-- Tag filter bar -->
    <div class="tag-bar">
      <button
        v-for="tag in allTags"
        :key="tag"
        class="tag-chip"
        :class="{ active: isActive( tag ) }"
        @click="toggleTag( tag )"
      >{{ tag }}</button>
      <button v-if="activeTags.length > 0" class="clear-chip" @click="clearTags">✕ clear</button>
    </div>

    <!-- Card grid -->
    <section class="card-grid">
      <a
        v-for="ex in cards"
        :key="ex.link"
        :href="withBase( ex.link )"
        class="card"
        :class="{ 'section-start': ex.section }"
      >
        <div v-if="ex.section" class="section-label" :title="ex.section.blurb">{{ ex.section.text }}</div>
        <div class="card-img-wrap">
          <img
            :src="imgSrc( ex.image )"
            :alt="ex.title"
            class="card-img"
            @error="e => e.target.src = PLACEHOLDER"
          />
        </div>
        <div class="card-body">
          <div class="card-title">{{ ex.title }}</div>
          <p class="card-desc">{{ ex.description }}</p>
          <div class="card-tags">
            <span
              v-for="tag in ex.tags"
              :key="tag"
              class="card-chip"
              :class="{ active: isActive( tag ) }"
              @click.prevent="toggleTag( tag )"
            >{{ tag }}</span>
          </div>
        </div>
      </a>
    </section>

    <div v-if="cards.length === 0" class="empty">
      No examples match the selected tags.
      <button @click="clearTags" class="clear-chip">✕ clear</button>
    </div>
  </div>
</template>

<style scoped>
/* ── Page ───────────────────────────────────────────── */
.gallery-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem 2rem 4rem;
}

.gallery-header { margin-bottom: 1.25rem; }
.gallery-header h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.3rem; }
.gallery-header p  { color: var(--vp-c-text-2); margin: 0; }

/* ── Tag bar ────────────────────────────────────────── */
.tag-bar {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin-bottom: 1.5rem;
}

.tag-chip {
  font-size: 0.78rem;
  padding: 0.25rem 0.7rem;
  border-radius: 99px;
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s, color 0.12s;
  user-select: none;
}
.tag-chip:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}
.tag-chip.active {
  background: var(--vp-c-brand-soft);
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

.clear-chip {
  font-size: 0.78rem;
  padding: 0.25rem 0.7rem;
  border-radius: 99px;
  border: 1px solid transparent;
  background: none;
  color: var(--vp-c-text-3);
  cursor: pointer;
  transition: color 0.12s;
}
.clear-chip:hover { color: var(--vp-c-text-1); }

/* ── Card grid ──────────────────────────────────────── */
.card-grid {
  display: grid;
  grid-template-columns: repeat( auto-fill, minmax( 210px, 1fr ) );
  gap: 2.1rem 1.25rem;   /* the row gap holds the section labels */
  padding-top: 1.6rem;
  overflow-x: clip;
}

.card {
  display: flex;
  flex-direction: column;
  border-radius: 10px;
  border: 1px solid var(--vp-c-divider);
  overflow: hidden;
  text-decoration: none;
  color: inherit;
  background: var(--vp-c-bg);
  transition: border-color 0.18s, box-shadow 0.18s, transform 0.18s;
}
.card:hover {
  border-color: var(--vp-c-brand-1);
  box-shadow: 0 6px 20px rgba( 0, 0, 0, 0.10 );
  transform: translateY( -2px );
}

.card-img-wrap {
  width: 100%;
  aspect-ratio: 4 / 3;
  overflow: hidden;
  background: var(--vp-c-bg-soft);
}
.card-img { width: 100%; height: 100%; object-fit: cover; display: block; }

.card-body {
  padding: 0.8rem;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  flex: 1;
}

.card-title { font-weight: 600; font-size: 0.9rem; color: var(--vp-c-text-1); }

.card-desc {
  font-size: 0.78rem;
  color: var(--vp-c-text-2);
  line-height: 1.45;
  flex: 1;
  margin: 0;
}

.card-tags { display: flex; flex-wrap: wrap; gap: 0.3rem; margin-top: 0.25rem; }

.card-chip {
  font-size: 0.67rem;
  padding: 0.15rem 0.5rem;
  border-radius: 99px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: background 0.12s, color 0.12s;
}
.card-chip:hover { background: var(--vp-c-bg-mute); }
.card-chip.active {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

/* ── Section labels, above the first card of each section ── */
.card { position: relative; overflow: visible; }
.card-img-wrap { border-radius: 10px 10px 0 0; }
.section-label {
  position: absolute;
  left: 0;
  width: 100vw;   /* the rule runs to the end of the row, clipped by the grid */
  bottom: calc( 100% + 0.35rem );
  padding-bottom: 0.2rem;
  border-bottom: 2px solid var(--vp-c-brand-1);
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--vp-c-brand-1);
  white-space: nowrap;
}

/* ── Empty state ────────────────────────────────────── */
.empty {
  grid-column: 1 / -1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.75rem;
  padding: 3rem;
  color: var(--vp-c-text-2);
  font-size: 0.9rem;
}
</style>
