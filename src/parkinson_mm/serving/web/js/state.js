// Objectif 1 : où en est le patient ? POST /state -> niveau estimé sans effet du traitement, vitesse, projection.
import { api } from './api.js';
import { axes, bindTip, frame, legend, linear, marker, niceMax, niceTicks, s, STATES, widthOf, years } from './chart.js';
import { editor, onChange, problem, request } from './dossier.js';
import { debounce, errorBox, fr, h, interval, pageHead, signed } from './dom.js';

let results = null, last = null;

export function visitMarks(svg, x, y, visits) {
  for (const v of visits) {
    const st = STATES[v.med_state];
    svg.append(marker(st.shape, x(v.time_years), y(v.np3), 5, st.color));
    svg.append(bindTip(s('circle', { class: 'hit', cx: x(v.time_years), cy: y(v.np3), r: 14 }), `${fr(v.np3, 0)} points`,
      `${st.label}, ${fr(v.time_years)} an(s) après l’inclusion`));
  }
}

function figure(r, visits, width) {
  const t0 = visits.at(-1).time_years;
  const proj = r.projection.map((p) => ({ ...p, t: t0 + p.months / 12 }));
  const top = Math.max(...visits.map((v) => v.np3), ...proj.map((p) => p.upper)) + 4;
  const f = frame(width, width < 640 ? 300 : 380, { top: 28, right: 24, bottom: 48, left: 44 }, 'Scores mesurés et niveau estimé');
  const x = linear([0, t0 + 2.2], [f.x0, f.x1]);
  const yAxis = niceMax(top);
  const y = linear([0, yAxis.max], [f.y0, f.y1]);
  f.svg.append(axes(f, x, y, { xTicks: niceTicks(0, t0 + 2, 6), yTicks: yAxis.ticks, fmtX: years,
    xTitle: 'Années depuis l’inclusion', yTitle: 'MDS-UPDRS III (points)' }));
  const band = [...proj.map((p) => `${x(p.t)},${y(p.upper)}`), ...proj.slice().reverse().map((p) => `${x(p.t)},${y(p.lower)}`)];
  f.svg.append(s('polygon', { points: band.join(' '), fill: 'var(--model)', 'fill-opacity': 0.08 }));
  f.svg.append(s('polyline', { points: proj.map((p) => `${x(p.t)},${y(p.level)}`).join(' '), fill: 'none',
    stroke: 'var(--model)', 'stroke-width': 2, 'stroke-linecap': 'round' }));
  f.svg.append(s('line', { x1: x(t0), x2: x(t0), y1: y(r.current_level_ci95[0]), y2: y(r.current_level_ci95[1]),
    stroke: 'var(--model)', 'stroke-width': 2 }));
  visitMarks(f.svg, x, y, visits);
  f.svg.append(marker('diamond', x(t0), y(r.current_level), 6, 'var(--model)'));
  const end = proj.at(-1);
  f.svg.append(s('text', { class: 'label', x: x(end.t) - 4, y: y(end.level) - 12, 'text-anchor': 'end' },
    `${fr(end.level)} dans 2 ans`));
  for (const p of proj) {
    f.svg.append(bindTip(s('circle', { class: 'hit', cx: x(p.t), cy: y(p.level), r: 12 }), `${fr(p.level)} ${interval(p.lower, p.upper)}`,
      `Niveau projeté dans ${p.months} mois (IC 95 %)`));
  }
  return h('figure', {}, f.svg,
    legend([...Object.values(STATES).map((st) => ({ label: st.label, color: st.color, shape: st.shape })),
      { label: 'Niveau estimé sans traitement, et sa projection (IC 95 %)', color: 'var(--model)', shape: 'diamond' }]),
    h('figcaption', {}, h('b', {}, 'Figure 3. '), `Le score mesuré varie d’une visite à l’autre et baisse sous traitement (ON).
      Le modèle mixte estime le niveau de la maladie dans une condition fixe, sans effet symptomatique du traitement,
      en tenant compte de toute la cohorte : un score ON plus bas n’est pas une amélioration de la maladie.`));
}

function readout(r) {
  return h('div', { class: 'readout' },
    h('div', {}, h('span', {}, 'Niveau actuel estimé'), h('strong', {}, fr(r.current_level)),
      h('small', {}, `IC 95 % ${interval(...r.current_level_ci95)}`)),
    h('div', {}, h('span', {}, 'Vitesse d’évolution'), h('strong', {}, `${signed(r.speed_pts_per_year)}`),
      h('small', {}, `points par an, IC 95 % ${interval(...r.speed_ci95)}`)),
    h('div', {}, h('span', {}, 'Dernier score mesuré'), h('strong', {}, fr(r.measured_last, 0)),
      h('small', {}, `sur ${r.n_visits} visite(s)`)));
}

async function compute() {
  if (!results || results.closest('.page')?.hidden) return;       // page cachée : calcul à l'ouverture
  const issue = problem();
  if (issue) { results.replaceChildren(errorBox(issue)); return; }
  results.classList.add('busy');
  try {
    const body = request();
    const r = await api('/state', body);
    last = { r, visits: body.visits };
    draw();
  } catch (error) {
    results.replaceChildren(errorBox(error.message));
  } finally { results.classList.remove('busy'); }
}

function draw() {
  if (!last) return;
  results.replaceChildren(readout(last.r), figure(last.r, last.visits, widthOf(results)));
}

const later = debounce(compute, 400);
onChange(() => later());

export function show(root) {
  results = h('div', { class: 'results', 'aria-live': 'polite' });
  root.replaceChildren(pageHead('Où en est-il ?', 'Objectif 1. À partir des visites passées, le modèle mixte estime le niveau '
    + 'actuel de la maladie et sa vitesse d’évolution, avec leur incertitude.'), h('div', { class: 'split' }, editor(), results));
  root.redraw = draw;
  compute();
}
