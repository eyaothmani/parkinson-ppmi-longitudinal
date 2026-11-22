// Navigation par ancre (#accueil, #actuel…), état des services de l'API, thème clair / sombre, redimensionnement.
import { api, setupKeyForm } from './api.js';
import * as assistant from './assistant.js';
import { hideTip, s } from './chart.js';
import { debounce, h, save, store } from './dom.js';
import * as factors from './factors.js';
import * as forecast from './forecast.js';
import * as home from './home.js';
import * as state from './state.js';

const PAGES = { accueil: home, actuel: state, futur: forecast, facteurs: factors, assistant };
const TITLES = { accueil: 'Vue d’ensemble', actuel: 'Où en est-il ?', futur: 'Où sera-t-il ?', facteurs: 'Qu’est-ce qui accélère ?',
  assistant: 'Assistant documentaire' };
const shown = new Set();

function current() { const id = location.hash.slice(1); return PAGES[id] ? id : 'accueil'; }

function go() {
  const id = current();
  hideTip();
  document.querySelectorAll('.page').forEach((p) => { p.hidden = p.dataset.page !== id; });
  document.querySelectorAll('.nav a').forEach((a) => {
    if (a.dataset.page === id) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current');
  });
  document.title = `${TITLES[id]}, Parkinson trajectoires PPMI`;
  const section = document.getElementById(`page-${id}`);
  if (!shown.has(id) || id === 'actuel' || id === 'futur') { PAGES[id].show(section); shown.add(id); }
  if (shown.size > 1) document.getElementById('main').focus({ preventScroll: true });
  window.scrollTo(0, 0);
}

function dot(ok) {
  return s('svg', { viewBox: '0 0 10 10', 'aria-hidden': 'true' },
    ok ? s('circle', { cx: 5, cy: 5, r: 4, fill: 'var(--rail-ink)' }) : s('circle', { cx: 5, cy: 5, r: 3.4, fill: 'none', stroke: 'var(--rail-muted)', 'stroke-width': 1.5 }));
}

async function services() {
  const list = document.getElementById('services');
  try {
    const health = await api('/health');
    const rows = [['Score actuel', health.state_model_loaded], ['Score futur', health.model_loaded], ['Facteurs', health.factors_loaded]];
    list.replaceChildren(...rows.map(([name, ok]) => h('li', {}, dot(ok), `${name} : ${ok ? 'prêt' : 'non chargé'}`)),
      h('li', {}, dot(true), 'Assistant : chargé à la première question'));
  } catch (error) {
    list.replaceChildren(h('li', {}, dot(false), error.message));
  }
}

function setupTheme() {
  const button = document.getElementById('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const apply = (theme) => {
    document.documentElement.dataset.theme = theme;
    button.textContent = theme === 'dark' ? 'Thème clair' : 'Thème sombre';
    save('ppmi-theme', theme);
  };
  apply(store('ppmi-theme', prefersDark ? 'dark' : 'light'));
  button.addEventListener('click', () => {
    apply(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  });
}

setupTheme();
setupKeyForm(() => { shown.clear(); services(); go(); });
window.addEventListener('hashchange', go);
window.addEventListener('resize', debounce(() => document.getElementById(`page-${current()}`).redraw?.(), 200));
services();
go();
