// Objectif 3 : qu'est-ce qui est associé à une progression plus rapide ? GET /factors (résultats agrégés).
// Graphique en forêt : différence de pente et IC 95 % ; rouge = plus rapide, bleu = plus lente, gris = non
// significatif après correction FDR. Associations observationnelles, jamais des causes.
import { api } from './api.js';
import { bindTip, frame, legend, linear, niceTicks, s, widthOf } from './chart.js';
import { errorBox, fr, h, interval, pageHead, signed } from './dom.js';

let all = null, root = null;
const filters = { onlySignificant: false, text: '' };

const colorOf = (f) => (!f.significatif_FDR ? 'var(--ref)' : f.effet_pente_pts_an > 0 ? 'var(--up)' : 'var(--down)');
const pValue = (p) => (p < 0.001 ? '< 0,001' : fr(p, 3));

function visible() {
  const text = filters.text.trim().toLowerCase();
  return all.factors
    .filter((f) => (!filters.onlySignificant || f.significatif_FDR) && f.facteur.toLowerCase().includes(text))
    .sort((a, b) => b.effet_pente_pts_an - a.effet_pente_pts_an);
}

function forest(rows, width) {
  const rowH = 28, labelW = Math.min(230, width * 0.36), valueW = width < 640 ? 0 : 150;
  const f = frame(width, rows.length * rowH + 52, { top: 6, right: valueW + 10, bottom: 44, left: labelW },
    'Différence de pente associée à chaque facteur');
  const m = Math.max(0.5, ...rows.map((r) => Math.max(Math.abs(r.IC95_bas), Math.abs(r.IC95_haut))));
  const x = linear([-m, m], [f.x0, f.x1]);
  const ticks = niceTicks(-m, m, 4);
  f.svg.append(s('g', { class: 'grid' }, ticks.map((t) => s('line', { x1: x(t), x2: x(t), y1: f.y1, y2: f.y0 }))));
  f.svg.append(s('line', { class: 'zero', x1: x(0), x2: x(0), y1: f.y1, y2: f.y0 }));
  f.svg.append(...ticks.map((t) => s('text', { class: 'tick-label', x: x(t), y: f.y0 + 18, 'text-anchor': 'middle' }, signed(t, 1))));
  f.svg.append(s('text', { class: 'axis-title', x: f.x0, y: f.y0 + 36 }, '← plus lente'),
    s('text', { class: 'axis-title', x: x(0), y: f.y0 + 36, 'text-anchor': 'middle' }, 'points par an'),
    s('text', { class: 'axis-title', x: f.x1, y: f.y0 + 36, 'text-anchor': 'end' }, 'plus rapide →'));
  rows.forEach((r, i) => {
    const cy = f.y1 + i * rowH + rowH / 2, color = colorOf(r);
    const name = r.facteur.length > 30 && width < 900 ? `${r.facteur.slice(0, 28)}…` : r.facteur;
    f.svg.append(s('text', { class: r.significatif_FDR ? 'label' : 'label-muted', x: 0, y: cy + 4 }, name));
    f.svg.append(s('line', { x1: x(r.IC95_bas), x2: x(r.IC95_haut), y1: cy, y2: cy, stroke: color, 'stroke-width': 2,
      'stroke-linecap': 'round' }));
    f.svg.append(s('circle', { cx: x(r.effet_pente_pts_an), cy, r: 5, fill: color, class: 'mark-ring' }));
    if (valueW) {
      f.svg.append(s('text', { class: 'tick-label', x: width - valueW + 12, y: cy + 4 },
        `${signed(r.effet_pente_pts_an, 2)} ${interval(r.IC95_bas, r.IC95_haut, 2)}`));
    }
    f.svg.append(bindTip(s('rect', { class: 'hit', x: 0, y: cy - rowH / 2, width, height: rowH }),
      `${signed(r.effet_pente_pts_an, 2)} pt/an ${interval(r.IC95_bas, r.IC95_haut, 2)}`,
      `${r.facteur} ; p corrigé (FDR) ${pValue(r.p_FDR)} ; ${r.n_patients} patients`));
  });
  return f.svg;
}

function table(rows) {
  const head = ['Facteur', 'Effet (pt/an)', 'IC 95 %', 'p corrigé (FDR)', 'Patients'];
  return h('details', { class: 'table-view' }, h('summary', {}, 'Voir le tableau des valeurs'),
    h('table', { class: 'data' }, h('thead', {}, h('tr', {}, head.map((c) => h('th', {}, c)))),
      h('tbody', {}, rows.map((r) => h('tr', {}, h('td', {}, r.facteur), h('td', { class: 'n' }, signed(r.effet_pente_pts_an, 2)),
        h('td', { class: 'n' }, interval(r.IC95_bas, r.IC95_haut, 2)), h('td', { class: 'n' }, pValue(r.p_FDR)),
        h('td', { class: 'n' }, r.n_patients))))));
}

function summary() {
  const significant = all.factors.filter((f) => f.significatif_FDR).length;
  return h('div', { class: 'readout' },
    h('div', {}, h('span', {}, 'Progression moyenne'), h('strong', {}, signed(all.mean_slope_pts_per_year, 2)),
      h('small', {}, 'points de MDS-UPDRS III par an')),
    h('div', {}, h('span', {}, 'Écart-type des pentes individuelles'), h('strong', {}, fr(all.sd_individual_slopes, 2)),
      h('small', {}, 'la vitesse varie beaucoup d’un patient à l’autre')),
    h('div', {}, h('span', {}, 'Facteurs associés après FDR'), h('strong', {}, `${significant} sur ${all.factors.length}`),
      h('small', {}, 'Benjamini-Hochberg, seuil 5 %')));
}

// Les filtres restent en place (le champ de recherche garde le focus) ; seul le graphique est redessiné.
function draw() {
  if (!all) return;
  const chart = root.querySelector('.chart'), rows = visible();
  chart.replaceChildren(
    rows.length ? h('figure', {}, forest(rows, widthOf(chart)),
      legend([{ label: 'Plus rapide', color: 'var(--up)' }, { label: 'Plus lente', color: 'var(--down)' },
        { label: 'Non significatif après FDR', color: 'var(--ref)' }]),
      h('figcaption', {}, h('b', {}, 'Figure 6. '), `Un modèle mixte par facteur (temps × facteur), ajusté sur l’âge, le sexe,
        la durée de la maladie et l’état de l’examen. Chaque point : différence de pente avec son IC à 95 %.
        Ce sont des associations observées dans PPMI, pas des causes.`))
      : h('p', { class: 'note' }, 'Aucun facteur ne correspond à ce filtre.'),
    table(rows));
}

export async function show(container) {
  root = container;
  const check = h('input', { type: 'checkbox', checked: filters.onlySignificant,
    onchange: (e) => { filters.onlySignificant = e.target.checked; draw(); } });
  const search = h('input', { type: 'search', class: 'search', placeholder: 'Chercher un facteur', value: filters.text,
    'aria-label': 'Chercher un facteur', oninput: (e) => { filters.text = e.target.value; draw(); } });
  const head = h('div', { class: 'summary' }, h('p', { class: 'note' }, 'Chargement des facteurs…'));
  root.replaceChildren(pageHead('Qu’est-ce qui accélère ?', 'Objectif 3. Quels facteurs cliniques, biologiques, génétiques '
    + 'et d’imagerie sont associés à une progression plus rapide ou plus lente du score moteur ?'),
  h('div', { class: 'results' }, head,
    h('div', { class: 'filters' }, h('label', { class: 'check' }, check, 'Seulement les facteurs significatifs après FDR'), search),
    h('div', { class: 'chart' })));
  root.redraw = draw;
  try { all = await api('/factors'); head.replaceChildren(summary()); draw(); } catch (error) {
    head.replaceChildren(errorBox(error.message));
  }
}
