<script setup>
import { ref, computed } from 'vue'
import { withBase } from 'vitepress'

const allExamples = [
  {
    title: 'Uniform Fill',
    description: 'OT plan from random Diracs to the uniform distribution on [0,1]². Visualizes the power diagram structure.',
    tags: [ 'ot-plan', '2d', 'basic' ],
    image: 'uniform-fill.png',
    link:  './uniform-fill',
  },
  {
    title: 'Image Transport',
    description: 'Use a grayscale image as target density. Transports a point cloud to match the image histogram.',
    tags: [ 'ot-plan', '2d', 'image', 'polynomial-grid' ],
    image: 'image-transport.png',
    link:  './image-transport',
  },
  {
    title: 'Lloyd Quantization',
    description: 'Iteratively move Diracs to their transport barycenters — converges to an optimal quantization of the target density.',
    tags: [ 'quantization', 'lloyd', '2d', 'meshing' ],
    image: 'lloyd-quantization.png',
    link:  './lloyd-quantization',
  },
  {
    title: 'Gaussian Mixture Fit',
    description: 'Minimize Wasserstein distance by gradient descent on Dirac positions. Demonstrates JAX and PyTorch autodiff.',
    tags: [ 'gradient', 'jax', 'torch', 'learning' ],
    image: 'gaussian-mixture-fit.png',
    link:  './gaussian-mixture-fit',
    tutorialLink: '../tutorials/gaussian-mixture-fit',
  },
  {
    title: 'Power Diagram Geometry',
    description: 'Inspect Laguerre cells: facets, simplex decomposition, distance from boundaries.',
    tags: [ 'geometry', 'cells', '2d' ],
    image: 'power-diagram-geometry.png',
    link:  './power-diagram-geometry',
  },
  {
    title: 'Periodic Metrics',
    description: 'OT and power diagrams with periodic boundary conditions (torus topology).',
    tags: [ 'geometry', 'periodic', '2d' ],
    image: 'periodic-metrics.png',
    link:  './periodic-metrics',
  },
  {
    title: 'Crowd Motion',
    description: 'Fokker-Planck equation as a Wasserstein gradient flow using Lagrangian (Moreau-Yosida) discretization.',
    tags: [ 'pde', 'gradient-flow', '2d', 'wasserstein' ],
    image: 'crowd-motion.png',
    link:  './crowd-motion',
  },
  {
    title: 'Incompressible Euler',
    description: 'Euler equations as a geodesic in the space of measure-preserving diffeomorphisms (Gallouët-Mérigot scheme).',
    tags: [ 'pde', 'euler', '3d', 'physics' ],
    image: 'euler-equations.png',
    link:  './euler-equations',
  },
  {
    title: 'Fokker-Planck Diffusion',
    description: 'Wasserstein gradient flow of the free energy functional — entropy plus potential energy.',
    tags: [ 'pde', 'diffusion', 'gradient-flow' ],
    image: 'fokker-planck.png',
    link:  './fokker-planck',
  },
]

// Placeholder image used for any missing image file
const PLACEHOLDER = withBase( '/examples/img/patate.png' )
function imgSrc( name ) {
  return withBase( `/examples/img/${name}` )
}

// All unique tags in a stable order
const allTags = computed( () => {
  const seen = new Set()
  allExamples.forEach( e => e.tags.forEach( t => seen.add( t ) ) )
  return [ ...seen ]
} )

// Active tags stored as a plain array for natural Vue 3 reactivity
const activeTags = ref( [] )

function toggleTag( tag ) {
  if ( activeTags.value.includes( tag ) ) {
    activeTags.value = activeTags.value.filter( t => t !== tag )
  } else {
    activeTags.value = [ ...activeTags.value, tag ]
  }
}

function clearTags() { activeTags.value = [] }
function isActive( tag ) { return activeTags.value.includes( tag ) }

const filtered = computed( () => {
  if ( activeTags.value.length === 0 ) return allExamples
  return allExamples.filter( e => e.tags.some( t => activeTags.value.includes( t ) ) )
} )
</script>

<template>
  <div class="gallery-page">
    <header class="gallery-header">
      <h1>Examples</h1>
      <p>Self-contained, runnable examples. Filter by tag to find what you need.</p>
    </header>

    <div class="gallery-body">
      <!-- Tag filter panel -->
      <aside class="tag-panel">
        <div class="panel-title">
          <span>Filter</span>
          <button v-if="activeTags.length > 0" class="clear-btn" @click="clearTags">Clear</button>
        </div>
        <label
          v-for="tag in allTags"
          :key="tag"
          class="tag-row"
          :class="{ active: isActive( tag ) }"
        >
          <input type="checkbox" :checked="isActive( tag )" @change="toggleTag( tag )" />
          <span>{{ tag }}</span>
        </label>
      </aside>

      <!-- Card grid -->
      <section class="card-grid">
        <a
          v-for="ex in filtered"
          :key="ex.title"
          :href="ex.link"
          class="card"
        >
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
                class="chip"
                :class="{ active: isActive( tag ) }"
                @click.prevent="toggleTag( tag )"
              >{{ tag }}</span>
            </div>
            <a v-if="ex.tutorialLink" :href="ex.tutorialLink" class="tutorial-link" @click.stop>
              Tutorial →
            </a>
          </div>
        </a>

        <div v-if="filtered.length === 0" class="empty">
          No examples match the selected tags.
          <button @click="clearTags" class="clear-btn">Clear filters</button>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
/* ── Page container ─────────────────────────────────── */
.gallery-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem 2rem 4rem;
}

.gallery-header {
  margin-bottom: 2rem;
}
.gallery-header h1 {
  font-size: 2rem;
  font-weight: 700;
  margin: 0 0 0.4rem;
}
.gallery-header p {
  color: var(--vp-c-text-2);
  margin: 0;
}

/* ── Two-column layout ──────────────────────────────── */
.gallery-body {
  display: flex;
  gap: 2rem;
  align-items: flex-start;
}

/* ── Tag filter panel ───────────────────────────────── */
.tag-panel {
  flex: 0 0 160px;
  position: sticky;
  top: 72px;
  background: var(--vp-c-bg-soft);
  border-radius: 10px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
}

.panel-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.78rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--vp-c-text-2);
  margin-bottom: 0.5rem;
}

.clear-btn {
  font-size: 0.75rem;
  color: var(--vp-c-brand-1);
  background: none;
  border: none;
  cursor: pointer;
  padding: 0;
}
.clear-btn:hover { text-decoration: underline; }

.tag-row {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  font-size: 0.83rem;
  cursor: pointer;
  padding: 0.25rem 0.4rem;
  border-radius: 5px;
  color: var(--vp-c-text-2);
  transition: background 0.12s, color 0.12s;
  user-select: none;
}
.tag-row:hover { background: var(--vp-c-bg-mute); }
.tag-row.active { color: var(--vp-c-brand-1); font-weight: 600; }
.tag-row input { accent-color: var(--vp-c-brand-1); }

/* ── Card grid ──────────────────────────────────────── */
.card-grid {
  flex: 1;
  display: grid;
  grid-template-columns: repeat( auto-fill, minmax( 210px, 1fr ) );
  gap: 1.25rem;
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
.card-img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.card-body {
  padding: 0.8rem;
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  flex: 1;
}

.card-title {
  font-weight: 600;
  font-size: 0.9rem;
  color: var(--vp-c-text-1);
}

.card-desc {
  font-size: 0.78rem;
  color: var(--vp-c-text-2);
  line-height: 1.45;
  flex: 1;
  margin: 0;
}

.card-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.3rem;
  margin-top: 0.25rem;
}

.chip {
  font-size: 0.67rem;
  padding: 0.15rem 0.5rem;
  border-radius: 99px;
  background: var(--vp-c-bg-soft);
  color: var(--vp-c-text-2);
  cursor: pointer;
  transition: background 0.12s, color 0.12s;
}
.chip:hover { background: var(--vp-c-bg-mute); }
.chip.active {
  background: var(--vp-c-brand-soft);
  color: var(--vp-c-brand-1);
  font-weight: 600;
}

.tutorial-link {
  font-size: 0.78rem;
  color: var(--vp-c-brand-1);
  text-decoration: none;
  margin-top: 0.2rem;
}
.tutorial-link:hover { text-decoration: underline; }

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

/* ── Responsive ─────────────────────────────────────── */
@media ( max-width: 700px ) {
  .gallery-body { flex-direction: column; }
  .tag-panel {
    flex: none;
    width: 100%;
    position: static;
    flex-direction: row;
    flex-wrap: wrap;
    gap: 0.5rem;
  }
  .panel-title { width: 100%; }
}
</style>
