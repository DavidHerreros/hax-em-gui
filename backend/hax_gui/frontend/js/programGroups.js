// Program groupings derived from the backend (single source of truth is
// hax_gui/adapters.py). Instead of hardcoding program names in each view, we
// fetch the derived flags once (`moded`, `latents_decoder`) and expose small
// predicates. New hax programs registered in adapters.py are picked up with no
// frontend edit. A fallback set covers the window before the fetch resolves (and
// offline), so synchronous callers (e.g. a job row's "viewable" check) stay safe.
import { api } from "./api.js";

// Kept in step with the current adapters.py registrations; only used until the
// live flags load, or if the request fails.
const FALLBACK = new Set(["hetsiren", "zernike3deep", "reconsiren", "reconsiren_het_only"]);

let _moded = null;            // Set<string> — programs with a train/predict --mode
let _predictLatents = null;   // Set<string> — predict run yields latents + a decoder

export async function loadProgramGroups(force = false) {
  if (_moded && !force) return;
  try {
    const list = await api.programs();
    _moded = new Set(list.filter((p) => p.moded).map((p) => p.name));
    _predictLatents = new Set(list.filter((p) => p.latents_decoder).map((p) => p.name));
  } catch { /* keep whatever we had (or the fallback via the getters) */ }
}

export function isModed(name) { return (_moded || FALLBACK).has(name); }
export function isPredictLatents(name) { return (_predictLatents || FALLBACK).has(name); }
