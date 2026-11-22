// Objectif 2 : où sera le patient ? POST /predict (score + intervalle à 80 %) et POST /explain (pourquoi ce score).
import { api } from './api.js';
import { axes, bindTip, frame, legend, linear, marker, niceMax, niceTicks, s, STATES, widthOf, years } from './chart.js';
import { editor, onChange, problem, request } from './dossier.js';
import { debounce, errorBox, fr, h, interval, pageHead, signed } from './dom.js';
import { visitMarks } from './state.js';

const choice = { horizon: 12, target: 'OFF' };
let results = null, body = null, last = null;

function trajectory(p, visits, width) {
  const t0 = visits.at(-1).time_years, now = visits.at(-1).np3, t1 = t0 + choice.horizon / 12;
  const top = Math.max(...visits.map((v) => v.np3), p.interval_80[1]) + 4;
  const f = frame(width, width < 640 ? 300 : 360, { top: 28, right: 120, bottom: 48, left: 44 }, 'Prédiction de la prochaine visite');
  const x = linear([0, t1 + 0.15], [f.x0, f.x1]);
  const yAxis = niceMax(top);
  const y = linear([0, yAxis.max], [f.y0, f.y1]);
  f.svg.append(axes(f, x, y, { xTicks: niceTicks(0, t1, 6), yTicks: yAxis.ticks, fmtX: years,
    xTitle: 'Années depuis l’inclusion', yTitle: 'MDS-UPDRS III (points)' }));
  f.svg.append(s('line', { x1: x(t0), x2: x(t1), y1: y(now), y2: y(now), stroke: 'var(--ref)', 'stroke-width': 2,
    'stroke-dasharray': '5 4' }));
  f.svg.append(s('line', { x1: x(t0), x2: x(t1), y1: y(now), y2: y(p.prediction), stroke: 'var(--model)',
    'stroke-width': 1.5, 'stroke-opacity': 0.45 }));
  f.svg.append(s('line', { x1: x(t1), x2: x(t1), y1: y(p.interval_80[0]), y2: y(p.interval_80[1]), stroke: 'var(--model)',
    'stroke-width': 2, 'stroke-linecap': 'round' }));
  visitMarks(f.svg, x, y, visits);
  f.svg.append(marker('diamond', x(t1), y(p.prediction), 6.5, 'var(--model)'));
  const apart = Math.abs(y(p.prediction) - y(now)) > 16;   // étiquettes séparées seulement si elles ne se chevauchent pas
  f.svg.append(s('text', { class: 'label', x: x(t1) + 14, y: y(p.prediction) + 4 }, `Prédiction ${fr(p.prediction)}`));
  if (apart) f.svg.append(s('text', { class: 'label-muted', x: x(t1) + 14, y: y(now) + 4 }, `Persistance ${fr(now, 0)}`));
  f.svg.append(bindTip(s('circle', { class: 'hit', cx: x(t1), cy: y(p.prediction), r: 16 }),
    `${fr(p.prediction)} ${interval(...p.interval_80)}`, `Score prédit dans ${choice.horizon} mois, examen ${STATES[choice.target].label}`));
  return h('figure', {}, f.svg,
    legend([{ label: 'Score prédit et intervalle à 80 %', color: 'var(--model)', shape: 'diamond' },
      { label: 'Persistance : « rien ne change »', color: 'var(--ref)', line: true, dash: true },
      ...Object.values(STATES).map((st) => ({ label: st.label, color: st.color, shape: st.shape }))]),
    h('figcaption', {}, h('b', {}, 'Figure 4. '), `Prédiction de la visite dans ${choice.horizon} mois, pour un examen
      ${STATES[choice.target].label}. L’état de l’examen à prédire fait partie des entrées du modèle : la prédiction est
      conditionnelle à cet état.`));
}

function explanation(e, width) {
  const used = e.effects.filter((g) => Math.abs(g.effect_points) >= 0.05);
  const unused = e.effects.filter((g) => Math.abs(g.effect_points) < 0.05).map((g) => g.group);
  const rowH = 32, labelW = Math.min(230, width * 0.4);
  const f = frame(width, used.length * rowH + 50, { top: 8, right: 70, bottom: 42, left: labelW }, 'Effet de chaque groupe de variables');
  const m = Math.ceil(Math.max(1, ...used.map((g) => Math.abs(g.effect_points))));
  const x = linear([-m, m], [f.x0, f.x1]);
  f.svg.append(s('line', { class: 'zero', x1: x(0), x2: x(0), y1: f.y1, y2: f.y0 }));
  f.svg.append(...niceTicks(-m, m, 4).map((t) => s('text', { class: 'tick-label', x: x(t), y: f.y0 + 18, 'text-anchor': 'middle' },
    signed(t, Number.isInteger(t) ? 0 : 1))));
  f.svg.append(s('text', { class: 'axis-title', x: f.x1, y: f.y0 + 36, 'text-anchor': 'end' }, 'Effet sur la prédiction (points)'));
  used.forEach((g, i) => {
    const cy = f.y1 + i * rowH + rowH / 2, v = g.effect_points, x0 = Math.min(x(0), x(v));
    f.svg.append(s('text', { class: 'label', x: 0, y: cy + 4 }, g.group));
    f.svg.append(s('rect', { x: x0, y: cy - 9, width: Math.max(2, Math.abs(x(v) - x(0))), height: 18, rx: 4,
      fill: v > 0 ? 'var(--up)' : 'var(--down)' }));
    f.svg.append(s('text', { class: 'label', x: v > 0 ? x(v) + 8 : x(v) - 8, y: cy + 4, 'text-anchor': v > 0 ? 'start' : 'end' }, signed(v)));
    f.svg.append(bindTip(s('rect', { class: 'hit', x: 0, y: cy - rowH / 2, width, height: rowH }), `${signed(v)} points`, g.group));
  });
  return h('figure', {}, f.svg,
    legend([{ label: 'Pousse vers un score plus élevé', color: 'var(--up)', shape: 'square' },
      { label: 'Pousse vers un score plus bas', color: 'var(--down)', shape: 'square' }]),
    h('figcaption', {}, h('b', {}, 'Figure 5. '), `Pour chaque groupe : écart entre la prédiction et ce qu’elle serait si ce
      groupe prenait la valeur typique des patients d’entraînement. Effets non additifs ; explication du modèle, pas des causes.`,
      unused.length ? ` Sans effet ici, car absents des entrées du modèle : ${unused.join(', ')}.` : ''));
}

function readout(p, e) {
  const version = /^\d+$/.test(p.model_version) ? `version ${p.model_version} du registre MLflow` : 'fichier local';
  return h('div', {},
    h('div', { class: 'readout' },
      h('div', {}, h('span', {}, `Score prédit dans ${choice.horizon} mois`), h('strong', {}, fr(p.prediction)),
        h('small', {}, `intervalle à 80 % ${interval(...p.interval_80)}`)),
      h('div', {}, h('span', {}, 'Par rapport à la dernière mesure'), h('strong', {}, signed(e.predicted_change)),
        h('small', {}, `dernière mesure : ${fr(e.current_score, 0)} points`))),
    h('p', { class: 'note' }, `Modèle servi : ${e.model}, ${version}.`));
}

async function compute() {
  if (!results || results.closest('.page')?.hidden) return;
  const issue = problem();
  if (issue) { body.replaceChildren(errorBox(issue)); return; }
  body.classList.add('busy');
  try {
    const input = { ...request(), horizon_months: choice.horizon, target_state: choice.target };
    const [p, e] = await Promise.all([api('/predict', input), api('/explain', input)]);
    last = { p, e, visits: input.visits };
    draw();
  } catch (error) {
    body.replaceChildren(errorBox(error.message));
  } finally { body.classList.remove('busy'); }
}

const later = debounce(compute, 400);
onChange(() => later());

function controls() {
  const out = h('output', {}, `${choice.horizon} mois`);
  const range = h('input', { type: 'range', min: 3, max: 24, step: 1, value: choice.horizon, 'aria-label': 'Délai en mois',
    oninput: (ev) => { choice.horizon = Number(ev.target.value); out.textContent = `${choice.horizon} mois`; later(); } });
  const states = h('div', { class: 'segmented', role: 'radiogroup', 'aria-label': 'État prévu à cette visite' },
    Object.entries(STATES).map(([value, st]) => h('label', {},
      h('input', { type: 'radio', name: 'cible', value, checked: choice.target === value,
        onchange: () => { choice.target = value; later(); } }), h('span', {}, st.label.split(' (')[0]))));
  return h('div', { class: 'controls' },
    h('label', { class: 'control' }, h('span', {}, 'Délai jusqu’à la visite : ', out), range),
    h('div', { class: 'control' }, h('span', {}, 'État prévu à cette visite'), states));
}

function draw() {
  if (!last) return;
  const width = widthOf(body);
  body.replaceChildren(readout(last.p, last.e), trajectory(last.p, last.visits, width),
    explanation(last.e, width), h('p', { class: 'note' }, last.p.disclaimer));
}

export function show(root) {
  body = h('div', { class: 'results', 'aria-live': 'polite' });
  results = h('div', { class: 'results' }, controls(), body);      // les réglages restent en place pendant le calcul
  root.replaceChildren(pageHead('Où sera-t-il ?', 'Objectif 2. Le modèle champion du registre prédit le score de la '
    + 'prochaine visite, avec un intervalle à 80 %, et le compare à la persistance : « rien ne change ».'),
  h('div', { class: 'split' }, editor(), results));
  root.redraw = draw;
  compute();
}
