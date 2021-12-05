import joplin from 'api';

// borrowed from backlinks plugin: https://github.com/ambrt/joplin-plugin-referencing-notes/blob/master/src/index.ts
function escapeTitleText(text: string) {
    return text.replace(/(\[|\])/g, '\\$1');
}

// the Favorites plugin does smt similar to what I envison wrt UI element
// (ie, it looks like the main note list in joplin)
//   https://emoji.discourse-cdn.com/twitter/house.png?v=10
export async function updateUIWithNoteList(panel, similar_notes) {
    const html_links = []
    for (const n of similar_notes) {
	const ahref = `<i>(${n.relative_score}%)</i> <a href="#" onclick="webviewApi.postMessage({type:'openNote',noteId:'${n.id}'})">${escapeTitleText(n.title)}</a>`
	html_links.push(ahref);
    }

    await updateHTML(panel, `${html_links.join('<br /><br />')}`);
}


// always keep title+scroll in html
export async function updateHTML(panel, html) {
    const titleHTML = '<h3>Semantically Similar Notes</h3>';
    
    // css overflow-y allows scrolling,
    //   needs height specified so we use 100% of viewport height
    //   todo this doesn't seem to work for the embedding computation text.
    //   so maybe vh is heigh of app window, not of webview panel?
    // todo: copy default joplin styling.
    //   (can this be programmatically deteremined?)
    const scrollStyleHTML = `
    		<style>
		.scroll_enabled {
		    overflow-y: auto;
		    max-height: 100vh;
		}
		.scroll_enabled::-webkit-scrollbar {
		    width: 15px;
		}
		.scroll_enabled::-webkit-scrollbar-corner {
		    background: rgba(0,0,0,0);
		}
		.scroll_enabled::-webkit-scrollbar-thumb {
		    background-color: #ccc;
		    border-radius: 6px;
		    border: 4px solid rgba(0,0,0,0);
		    background-clip: content-box;
		    min-width: 32px;
		    min-height: 32px;
		}
		.scroll_enabled::-webkit-scrollbar-track {
		    background-color: rgba(0,0,0,0);
		}
		</style>
    `;
    
    await joplin.views.panels.setHtml(panel, titleHTML + scrollStyleHTML +
				             `<div class="scroll_enabled">` +
 				             html +
				             `</div>`);
}
