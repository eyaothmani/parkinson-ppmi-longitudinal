// Seule porte vers les données : les routes de l'API. Clé d'API facultative (si la variable API_KEY est définie
// côté serveur) : demandée une fois, gardée pour la session du navigateur seulement.
import { save, store } from './dom.js';

const KEY = 'ppmi-api-key';

export class ApiError extends Error {
  constructor(message, status) { super(message); this.status = status; }
}

function readable(detail, status) {
  if (Array.isArray(detail) && detail.length) {         // erreurs de validation (422) de FastAPI
    const first = detail[0];
    const where = (first.loc || []).filter((p) => typeof p === 'string' && p !== 'body').join(' › ');
    return `Saisie refusée par l'API${where ? ` (${where})` : ''} : ${first.msg}`;
  }
  if (typeof detail === 'string') return detail;
  return `L'API a répondu ${status}.`;
}

export async function api(path, body) {
  const headers = { 'Content-Type': 'application/json' };
  const key = store(KEY, '');
  if (key) headers['X-API-Key'] = key;
  let response;
  try {
    response = await fetch(path, body ? { method: 'POST', headers, body: JSON.stringify(body) } : { headers });
  } catch {
    throw new ApiError("L'API ne répond pas. Lancez-la avec `make serve` (ou uvicorn), puis rechargez la page.", 0);
  }
  if (response.status === 401) {
    document.getElementById('keyform').hidden = false;
    throw new ApiError("Cette API demande une clé : saisissez-la dans la barre latérale.", 401);
  }
  if (!response.ok) {
    let detail;
    try { detail = (await response.json()).detail; } catch { /* corps vide */ }
    throw new ApiError(readable(detail, response.status), response.status);
  }
  return response.json();
}

export function setupKeyForm(onSaved) {
  const form = document.getElementById('keyform');
  form.addEventListener('submit', (event) => {
    event.preventDefault();
    save(KEY, document.getElementById('apikey').value.trim());
    form.hidden = true;
    onSaved();
  });
}
