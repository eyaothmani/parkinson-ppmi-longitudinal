// Dossier FICTIF saisi à la main, partagé par les objectifs 1 et 2. Il reste dans l'onglet du navigateur
// (sessionStorage) et n'est envoyé qu'à l'API locale : aucune donnée PPMI n'entre ici.
import { h, save, store } from './dom.js';

const KEY = 'ppmi-dossier-fictif';
const EXAMPLE = {
  patient: { age_bl: 61, male: 1, disease_dur_bl: 0.8 },
  visits: [
    { time_years: 0, np3: 18, med_state: 'untreated' },
    { time_years: 1, np3: 22, med_state: 'untreated' },
    { time_years: 2, np3: 31, med_state: 'OFF' },
    { time_years: 3, np3: 24, med_state: 'ON' },
  ],
};
export const dossier = store(KEY, structuredClone(EXAMPLE));
const listeners = new Set();

export function onChange(fn) { listeners.add(fn); }

function changed() { save(KEY, dossier); listeners.forEach((fn) => fn()); }

const num = (v) => (v === '' || v == null ? NaN : Number(v));

export function problem() {
  const v = dossier.visits;
  if (!v.length) return 'Ajoutez au moins une visite.';
  if (v.some((x) => !Number.isFinite(num(x.time_years)) || !Number.isFinite(num(x.np3)))) {
    return 'Chaque visite a besoin d’un temps (en années) et d’un score MDS-UPDRS III.';
  }
  if (v.some((x) => num(x.np3) < 0 || num(x.np3) > 132)) return 'Le MDS-UPDRS III va de 0 à 132.';
  if (v.some((x, i) => i && num(x.time_years) <= num(v[i - 1].time_years))) {
    return 'Les visites doivent être dans l’ordre du temps, sans deux visites au même moment.';
  }
  return null;
}

export function request() {
  const p = dossier.patient;
  const opt = (x) => (Number.isFinite(num(x)) ? num(x) : null);
  return {
    visits: dossier.visits.map((x) => ({ time_years: num(x.time_years), np3: num(x.np3), med_state: x.med_state })),
    patient: { age_bl: opt(p.age_bl), male: p.male === '' ? null : Number(p.male), disease_dur_bl: opt(p.disease_dur_bl) },
  };
}

const STATE_OPTIONS = [['untreated', 'Non traité'], ['OFF', 'OFF'], ['ON', 'ON']];
const trash = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 4h10M6.5 4V2.5h3V4M4.5 4l.7 9.5h5.6L11.5 4"/></svg>';

function row(visit, i, rerender) {
  const set = (key) => (e) => { visit[key] = e.target.value; changed(); };
  const remove = h('button', { type: 'button', class: 'icon-btn', 'aria-label': `Supprimer la visite ${i + 1}`,
    onclick: () => { dossier.visits.splice(i, 1); changed(); rerender(); } });
  remove.innerHTML = trash;                               // icône fixe, écrite ici : pas une donnée
  const state = h('select', { 'aria-label': `État à la visite ${i + 1}`, onchange: set('med_state') },
    STATE_OPTIONS.map(([value, label]) => h('option', { value, selected: visit.med_state === value }, label)));
  return h('tr', {},
    h('td', {}, h('input', { type: 'number', step: '0.5', min: 0, max: 40, value: visit.time_years,
      'aria-label': `Années depuis l’inclusion, visite ${i + 1}`, oninput: set('time_years') })),
    h('td', {}, h('input', { type: 'number', step: 1, min: 0, max: 132, value: visit.np3,
      'aria-label': `MDS-UPDRS III, visite ${i + 1}`, oninput: set('np3') })),
    h('td', {}, state), h('td', {}, remove));
}

function field(label, control) { return h('label', { class: 'field' }, label, control); }

export function editor() {
  const body = h('tbody');
  const rerender = () => body.replaceChildren(...dossier.visits.map((v, i) => row(v, i, rerender)));
  rerender();
  const p = dossier.patient;
  const setP = (key) => (e) => { p[key] = e.target.value; changed(); };
  const add = () => {
    const last = dossier.visits.at(-1);
    dossier.visits.push({ time_years: last ? Number(last.time_years) + 1 : 0, np3: last ? last.np3 : 20,
      med_state: last ? last.med_state : 'untreated' });
    changed(); rerender();
  };
  return h('section', { class: 'bench', 'aria-label': 'Dossier fictif' },
    h('h2', {}, 'Dossier fictif'),
    h('p', { class: 'note' }, 'Un patient inventé, modifiable. Il reste dans cet onglet et n’est envoyé qu’à l’API locale.'),
    h('table', { class: 'visits' },
      h('thead', {}, h('tr', {}, h('th', {}, 'Années'), h('th', {}, 'Score'), h('th', {}, 'État'), h('th', {}))), body),
    h('button', { type: 'button', class: 'btn secondary', onclick: add }, 'Ajouter une visite'),
    h('div', { class: 'fields' },
      field('Âge à l’inclusion', h('input', { type: 'number', min: 18, max: 110, value: p.age_bl, oninput: setP('age_bl') })),
      field('Sexe', h('select', { onchange: setP('male') },
        [['1', 'Homme'], ['0', 'Femme'], ['', 'Non précisé']].map(([value, label]) =>
          h('option', { value, selected: String(p.male ?? '') === value }, label)))),
      field('Années depuis le diagnostic', h('input', { type: 'number', min: 0, max: 40, step: 0.1,
        value: p.disease_dur_bl, oninput: setP('disease_dur_bl') }))));
}
