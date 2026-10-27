import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'
import path from 'node:path'
import { gallerySidebar, GALLERY_DIRS } from './gallery.js'

// in dev, refresh the 'Examples' sidebar when an example or a guide page is added, removed or its frontmatter changes
function examplesSidebarPlugin() {
  let siteConfig: any
  return {
    name: 'sdot-examples-sidebar',
    configResolved( config: any ) { siteConfig = config.vitepress },
    configureServer( server: any ) {
      const refresh = ( file: string ) => {
        if ( !file.endsWith( '.md' ) || !GALLERY_DIRS.some( d => file.startsWith( d + path.sep ) ) ) return
        const sidebar = siteConfig?.site?.themeConfig?.sidebar
        if ( !sidebar ) return
        const items = gallerySidebar()
        if ( JSON.stringify( items ) === JSON.stringify( sidebar[ '/examples/' ] ) ) return
        sidebar[ '/examples/' ] = items
        const mod = server.moduleGraph.getModuleById( '/@siteData' )
        if ( mod ) server.moduleGraph.invalidateModule( mod )
        server.ws.send( { type: 'full-reload' } )
      }
      server.watcher.on( 'add', refresh )
      server.watcher.on( 'unlink', refresh )
      server.watcher.on( 'change', refresh )
    },
  }
}

export default defineConfig({
  title: "SDOT",
  description: "Fast, differentiable, N-dimensional Semi-Discrete Optimal Transport — with JAX and PyTorch.",
  base: '/sdot/',
  markdown: {
    config: ( md ) => { md.use( mathjax3 ) }
  },
  vite: {
    plugins: [ examplesSidebarPlugin() ]
  },
  vue: {
    template: {
      compilerOptions: {
        isCustomElement: ( tag ) => tag.startsWith( 'mjx-' )
      }
    }
  },
  themeConfig: {
    nav: [
      { text: 'Guide',      link: '/guide/getting-started' },
      { text: 'Examples',   link: '/examples/' },
      { text: 'API',        link: '/api/ot-plan' },
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Guide',
          items: [
            { text: 'Getting Started',          link: '/guide/getting-started' },
            { text: 'Plans and Settings',       link: '/guide/ot-plan' },
            { text: 'Distributions',            link: '/guide/distributions/', collapsed: false, items: [
              { text: 'Convex Polytopes',       link: '/guide/distributions/polytope' },
              { text: 'Images and Splines',     link: '/guide/distributions/image' },
              { text: 'Sums of Gaussians',      link: '/guide/distributions/gaussians' },
              { text: 'Meshes',                 link: '/guide/distributions/mesh' },
              { text: 'Lines and Surfaces',     link: '/guide/distributions/codim-mesh' },
              { text: 'Projected Sources',      link: '/guide/distributions/projected' },
            ] },
            { text: 'Ground Metrics',           link: '/guide/ground-metrics/', collapsed: false, items: [
              { text: 'Periodicity',            link: '/guide/ground-metrics/periodicity' },
              { text: 'Partial Transport',      link: '/guide/ground-metrics/partial' },
            ] },
            { text: 'Duplicates and Close Seeds', link: '/guide/duplicates' },
            { text: 'The Cell Complex',         link: '/guide/cell-complex' },
            { text: 'Frameworks and Devices',   link: '/guide/backends' },
            { text: 'Performance',              link: '/guide/performance' },
            { text: 'Background',               link: '/guide/background' },
            { text: 'Roadmap',                  link: '/guide/roadmap' },
            { text: 'The Visualizer',           link: '/guide/visualizer' },
          ]
        }
      ],
      '/examples/': gallerySidebar(),
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'W2 and Plans',   link: '/api/ot-plan' },
            { text: 'Distributions',  link: '/api/distributions' },
            { text: 'Power Diagrams', link: '/api/power-diagram' },
            { text: 'Visualization',  link: '/api/visualization' },
          ]
        }
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/sdot-team/sdot' }
    ],

    search: { provider: 'local' },

    editLink: {
      pattern: 'https://github.com/sdot-team/sdot/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    },

    footer: {
      message: 'SDOT — H. Leclerc, Q. Mérigot, T. Gallouët · LMO / INRIA PARMA',
    }
  }
})
