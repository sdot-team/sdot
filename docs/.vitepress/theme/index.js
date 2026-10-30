import DefaultTheme from 'vitepress/theme'
import ExamplesGallery from './ExamplesGallery.vue'
import BenchTable from './BenchTable.vue'

export default {
  extends: DefaultTheme,
  enhanceApp( { app } ) {
    app.component( 'ExamplesGallery', ExamplesGallery )
    app.component( 'BenchTable', BenchTable )
  }
}
