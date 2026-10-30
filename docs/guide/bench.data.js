// The results of `bench/page.py`: one file per machine in `bench/` ( `<name>-quick.json`, a smoke run, is left out ).
import fs from 'node:fs'
import path from 'node:path'

export default {
  watch: [ './bench/*.json' ],
  load( files ) {
    return files
      .filter( f => !f.endsWith( '-quick.json' ) )
      .sort()
      .map( f => JSON.parse( fs.readFileSync( f, 'utf-8' ) ) )
  }
}
