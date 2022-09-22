// originally copied from note graph ui plugin

import joplin from 'api';
import { SettingItemType } from 'api/types';

const DEFAULT_BATCH_SIZE = 100;

enum TFJS_BACKEND_ENUM {
  wasm = 'WebAssembly',
  webgl = 'WebGL',
  cpu = 'CPU',
}

export async function registerSettings() {
  const sectionName = "semantically-similar.settings"
  await joplin.settings.registerSection(sectionName, {
    label: 'Semantically Similar Notes',
    // Check out https://forkaweso.me/Fork-Awesome/icons/ for available icons.
    iconName: 'fas fa-lightbulb'
  });

  await joplin.settings.registerSettings({
    SETTING_TFJS_BACKEND: {
      value: "wasm", // this is defining the default value
      type: SettingItemType.String,
      section: sectionName,
      isEnum: true,
      public: true,
      label: 'Tensorflow Backend',
      description: 'WebGL can be like 25-50x faster than CPU. Needs restart to affect initial embedding computation.',

      // todo use enum keys, too?
      options: {
	wasm: TFJS_BACKEND_ENUM.wasm,
        webgl: TFJS_BACKEND_ENUM.webgl,
        cpu: TFJS_BACKEND_ENUM.cpu,
      }
    },
    SETTING_BATCH_SIZE: {
      value: DEFAULT_BATCH_SIZE,
      type: SettingItemType.Int,
      section: sectionName,
      public: true,
      label: 'Model Batch Size (restart)',
      description: '# notes in input to model at one time. Really only affects speed of initial embeddings computation. Restart after changing.'
    }
  });
}

export async function getSelectedBackend(): Promise<TFJS_BACKEND_ENUM> {
  const tfjsBackendSelection = await joplin.settings.value("SETTING_TFJS_BACKEND");
  return tfjsBackendSelection as TFJS_BACKEND_ENUM;

}
