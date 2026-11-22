// Vue d'ensemble. Figure 1 : l'éventail des trajectoires, tracé avec les vrais chiffres de la cohorte (pente moyenne ± un écart-type
// des pentes individuelles). Figure 2 : erreur à la visite suivante, chaque modèle face aux deux références.
import { api } from './api.js';
import { axes, bindTip, frame, legend, linear, marker, niceTicks, s, widthOf } from './chart.js';
import { errorBox, fr, h, pageHead, signed } from './dom.js';

let data = null;

function fan(stats, width) {
  const m = stats.mean_slope_pts_per_year, sd = stats.sd_individual_slopes, years = 5;
  const lines = [
    { slope: m + sd, color: 'var(--up)', name: 'Rapide : pente moyenne + 1 écart-type', cls: 'd3' },
    { slope: m, color: 'var(--ink)', name: 'Pente moyenne de la cohorte', cls: 'd2' },
    { slope: Math.max(m - sd, 0), color: 'var(--down)', name: 'Lente : pente moyenne − 1 écart-type', cls: '' },
  ];
  const height = width < 640 ? 300 : 440;
  const f = frame(width, height, { top: 16, right: 92, bottom: 44, left: 14 }, 'Éventail des trajectoires de la cohorte');
  const x = linear([0, years], [f.x0, f.x1]);
  const top = Math.ceil(((m + sd) * years) / 5) * 5;
  const y = linear([0, top], [f.y0, f.y1]);
  f.svg.append(s('line', { class: 'zero', x1: f.x0, x2: f.x1, y1: f.y0, y2: f.y0 }));
  f.svg.append(...[0, 1, 2, 3, 4, 5].map((t) => s('text', { class: 'tick-label', x: x(t), y: f.y0 + 18,
    'text-anchor': 'middle' }, t === 0 ? 'inclusion' : `${t} an${t > 1 ? 's' : ''}`)));
  f.svg.append(s('polygon', { fill: 'var(--paper-2)', 'fill-opacity': 0.8, points:
    `${x(0)},${y(0)} ${x(years)},${y(lines[0].slope * years)} ${x(years)},${y(lines[2].slope * years)}` }));
  for (const line of lines) {
    const end = line.slope * years;
    f.svg.append(s('line', { class: `fan-line fan-draw ${line.cls}`, pathLength: 1, x1: x(0), y1: y(0), x2: x(years),
      y2: y(end), stroke: line.color }));
    f.svg.append(marker('circle', x(years), y(end), 5, line.color));
    f.svg.append(s('text', { class: 'label', x: x(years) + 12, y: y(end) + 4 }, `${signed(line.slope, 1)} pt/an`));
    f.svg.append(bindTip(s('circle', { class: 'hit', cx: x(years), cy: y(end), r: 16 }),
      `${signed(end, 0)} points en 5 ans`, line.name));
  }
  f.svg.append(s('circle', { cx: x(0), cy: y(0), r: 7, fill: 'var(--ink)' }));
  return h('figure', {}, f.svg, legend(lines.map((l) => ({ label: l.name, color: l.color, line: true }))),
    h('figcaption', {}, h('b', {}, 'Figure 1. '), `Variation du MDS-UPDRS III depuis l’inclusion, modèle mixte linéaire
      sur ${data.state?.patients?.toLocaleString('fr-FR') ?? '1 512'} patients (ajusté sur l’âge, le sexe, la durée et l’état de l’examen).
      Environ deux patients sur trois ont une pente dans la zone colorée, si les pentes suivent une loi normale.`));
}

function questions(factors) {
  const best = Object.entries(data.cv?.cv ?? {}).filter(([k]) => k !== 'Persistance').sort((a, b) => a[1].MAE - b[1].MAE)[0];
  const gain = best && data.reference_etat ? data.reference_etat.cv_MAE - best[1].MAE : null;
  const significant = factors.factors.filter((f) => f.significatif_FDR).length;
  const block = (title, value, text, href, action) => h('article', { class: 'question' },
    h('h3', {}, title), h('p', { class: 'value' }, value), h('p', {}, text), h('a', { href }, action));
  return h('section', { class: 'questions', 'aria-label': 'Les trois objectifs' },
    block('Où en est-il ?', data.state ? `${fr(data.state.test.MAE)} pts` : '—',
      'd’erreur moyenne du score d’état sur des patients jamais vus. Le modèle mixte sépare le niveau de la maladie de l’effet du traitement.',
      '#actuel', 'Estimer un niveau'),
    block('Où sera-t-il ?', gain == null ? '—' : `${fr(gain, 2)} pt`,
      'de mieux que la référence juste (persistance ajustée à l’état), en validation croisée. Le deep learning fait jeu égal avec les modèles simples.',
      '#futur', 'Prédire la prochaine visite'),
    block('Qu’est-ce qui accélère ?', `${significant} sur ${factors.factors.length}`,
      'facteurs associés à la vitesse de progression après correction de Benjamini-Hochberg. Des associations, pas des causes.',
      '#facteurs', 'Voir les facteurs'));
}

function ladderRows() {
  const cv = data.cv?.cv ?? {}, rows = [];
  if (cv.Persistance) rows.push({ name: 'Persistance brute', mae: cv.Persistance.MAE, kind: 'ref' });
  if (data.reference_etat) rows.push({ name: 'Persistance ajustée à l’état', mae: data.reference_etat.cv_MAE, kind: 'ref' });
  if (data.etat_cv) rows.push({ name: 'Score d’état (modèle mixte)', mae: data.etat_cv.cv_MAE, kind: 'model' });
  const short = (name) => name.split(' ')[0];            // « GRU – trajectoire (M1) » -> « GRU »
  for (const [name, mae] of Object.entries(data.sensibilite?.cv_MAE ?? {})) {
    if (name.includes('sans état cible')) rows.push({ name: `${short(name)} sans l’état cible`, mae, kind: 'blind' });
  }
  for (const [name, v] of Object.entries(cv)) if (name !== 'Persistance') rows.push({ name: short(name), mae: v.MAE, kind: 'model' });
  return rows.sort((a, b) => b.mae - a.mae);
}

const KIND = {
  ref: { label: 'Référence', color: 'var(--ref)', hollow: false },
  blind: { label: 'Modèle privé de l’état de l’examen à prédire', color: 'var(--model)', hollow: true },
  model: { label: 'Modèle', color: 'var(--model)', hollow: false },
};

function ladder(width) {
  const rows = ladderRows();
  if (!rows.length) return null;
  const labelW = Math.min(250, width * 0.42), rowH = 34;
  const f = frame(width, rows.length * rowH + 56, { top: 10, right: 64, bottom: 46, left: labelW },
    'Erreur à la visite suivante par modèle');
  const lo = Math.floor(Math.min(...rows.map((r) => r.mae)) * 5) / 5 - 0.1, hi = Math.ceil(Math.max(...rows.map((r) => r.mae)) * 5) / 5;
  const x = linear([lo, hi], [f.x0, f.x1]);
  const ticks = niceTicks(lo, hi, 5);
  f.svg.append(s('g', { class: 'grid' }, ticks.map((t) => s('line', { x1: x(t), x2: x(t), y1: f.y1, y2: f.y0 }))));
  f.svg.append(...ticks.map((t) => s('text', { class: 'tick-label', x: x(t), y: f.y0 + 18, 'text-anchor': 'middle' }, fr(t, 1))));
  f.svg.append(s('text', { class: 'axis-title', x: f.x1, y: f.y0 + 38, 'text-anchor': 'end' }, 'Erreur moyenne (points, plus bas = mieux)'));
  rows.forEach((r, i) => {
    const cy = f.y1 + i * rowH + rowH / 2, k = KIND[r.kind];
    f.svg.append(s('text', { class: r.kind === 'ref' ? 'label-muted' : 'label', x: 0, y: cy + 4 }, r.name));
    f.svg.append(s('line', { x1: f.x0, x2: x(r.mae), y1: cy, y2: cy, stroke: 'var(--rule)', 'stroke-width': 1 }));
    f.svg.append(marker('circle', x(r.mae), cy, 6, k.color, k.hollow));
    f.svg.append(s('text', { class: 'label', x: f.x1 + 12, y: cy + 4 }, fr(r.mae, 2)));
    f.svg.append(bindTip(s('rect', { class: 'hit', x: 0, y: cy - rowH / 2, width, height: rowH }), `${fr(r.mae, 2)} points`,
      `${r.name} (${k.label.toLowerCase()})`));
  });
  return h('figure', {}, f.svg,
    legend(Object.values(KIND).reverse().map((k) => ({ label: k.label, color: k.color, hollow: k.hollow }))),
    h('figcaption', {}, h('b', {}, 'Figure 2. '), `Erreur moyenne à la visite suivante, en validation croisée sur les patients
      de développement. Les modèles connaissent l’état ON/OFF de l’examen à prédire : la persistance ajustée, qui le connaît
      aussi, est la référence juste. Privés de cet état, ils gardent l’essentiel de leur avance.`));
}

export async function show(root) {
  root.replaceChildren(h('p', { class: 'note' }, 'Chargement des résultats…'));
  try {
    const [summary, factors] = await Promise.all([api('/summary'), api('/factors')]);
    data = summary;
    draw(root, factors);
  } catch (error) {
    root.replaceChildren(pageHead('Trajectoires de la maladie de Parkinson', 'Cohorte PPMI.'), errorBox(error.message));
  }
}

function draw(root, factors) {
  const width = widthOf(root);
  const hero = h('section', { class: 'hero' },
    pageHead('Même point de départ, trajectoires différentes.',
      `Dans PPMI, le score moteur augmente en moyenne de ${fr(factors.mean_slope_pts_per_year)} point par an, mais la vitesse
       varie beaucoup d’un patient à l’autre. Ce projet mesure où en est un patient, prévoit sa prochaine visite et cherche
       ce qui est associé à une progression plus rapide.`),
    fan(factors, width));
  root.replaceChildren(hero,
    h('h2', { class: 'section' }, 'Trois questions, une cohorte'), questions(factors),
    h('h2', { class: 'section' }, 'Lire les erreurs avec la bonne référence'), ladder(width),
    h('p', { class: 'note' }, `Outil de recherche : les patients des pages suivantes sont fictifs, saisis à la main, et une
      erreur moyenne de six points dépasse la différence cliniquement importante (3,25 points). Aucun usage individuel.`));
  root.redraw = () => draw(root, factors);
}
