from api/joplin.d.ts, it says we can't bundle native pkgs with a plugin bc of cross-platform. does this mean i can't use tensorflow with native cpp bindings? and need pure js version?

from api/joplinplugins.d.ts, i think i can store the LM in dataDir() loc

use joplinviewspanels.d.ts to create View to display list of semantically similar notes

workspace service to ge currently selected note, and when note content changes

package.json implies joplin uses webpack for building dist (informs tensorlfow.js approach)