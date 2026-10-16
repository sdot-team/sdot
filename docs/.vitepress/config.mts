import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'
import fs from 'node:fs'
import path from 'node:path'

const examplesDir = path.resolve( __dirname, '../examples' )

// the 'Examples' sidebar, from the frontmatter of examples/*.md -- same filter and order as examples/examples.data.js
function examplesSidebarItems() {
  const examples = fs.readdirSync( examplesDir )
    .filter( f => f.endsWith( '.md' ) && f !== 'index.md' )
    .map( f => {
      const slug = f.slice( 0, -3 )
      const fm = fs.readFileSync( path.join( examplesDir, f ), 'utf8' ).match( /^---\r?\n([\s\S]*?)\r?\n---/ )?.[ 1 ] ?? ''
      const field = ( key: string ) => fm.match( new RegExp( `^${ key }:\\s*(.*?)\\s*$`, 'm' ) )?.[ 1 ].replace( /^(['"])(.*)\1$/, '$2' )
      return {
        title: field( 'title' ) || slug,
        order: Number( field( 'order' ) ?? 999 ),
        draft: field( 'draft' ) === 'true',
        link:  `/examples/${ slug }`,
      }
    } )
    .filter( e => !e.draft )
    .sort( ( a, b ) => a.order - b.order || a.title.localeCompare( b.title ) )
  return [
    { text: 'Gallery', link: '/examples/' },
    ...examples.map( e => ( { text: e.title, link: e.link } ) ),
  ]
}

// in dev, refresh the 'Examples' sidebar when an example is added, removed or its frontmatter changes
function examplesSidebarPlugin() {
  let siteConfig: any
  return {
    name: 'sdot-examples-sidebar',
    configResolved( config: any ) { siteConfig = config.vitepress },
    configureServer( server: any ) {
      const refresh = ( file: string ) => {
        if ( path.dirname( file ) !== examplesDir || !file.endsWith( '.md' ) ) return
        const group = siteConfig?.site?.themeConfig?.sidebar?.[ '/examples/' ]?.[ 0 ]
        if ( !group ) return
        const items = examplesSidebarItems()
        if ( JSON.stringify( items ) === JSON.stringify( group.items ) ) return
        group.items = items
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
            { text: 'W2, Plans and Settings',   link: '/guide/ot-plan' },
            { text: 'Distributions',            link: '/guide/distributions/', collapsed: false, items: [
              { text: 'Lines and Surfaces',     link: '/guide/distributions/codim-mesh' },
            ] },
            { text: 'Ground Metrics',           link: '/guide/ground_metrics' },
            { text: 'Duplicates and Close Seeds', link: '/guide/duplicates' },
            { text: 'Frameworks and Devices',   link: '/guide/backends' },
            { text: 'Performance',              link: '/guide/performance' },
            { text: 'Background',               link: '/guide/background' },
            { text: 'Roadmap',                  link: '/guide/roadmap' },
          ]
        }
      ],
      '/examples/': [
        {
          text: 'Examples',
          items: examplesSidebarItems()
        }
      ],
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
