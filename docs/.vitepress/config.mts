import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

export default defineConfig({
  title: "SDOT",
  description: "Fast, differentiable, N-dimensional Semi-Discrete Optimal Transport — with JAX and PyTorch.",
  base: '/sdot/',
  markdown: {
    config: ( md ) => { md.use( mathjax3 ) }
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
            { text: 'Distributions',            link: '/guide/distributions' },
            { text: 'Differentiability',        link: '/guide/differentiability' },
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
          items: [
            { text: 'Gallery',               link: '/examples/' },
            { text: 'Lloyd Quantization',    link: '/examples/lloyd-quantization' },
            { text: 'Gaussian Mixture Fit',  link: '/examples/gaussian-mixture-fit' },
          ]
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
