// Graphiques en SVG écrits à la main (aucune bibliothèque) : échelles, axes, marqueurs, légende, infobulle.
// Règles : traits de 2 px, marqueurs ≥ 8 px avec un anneau de la couleur du fond, grille fine et discrète,
// texte toujours dans les couleurs de texte (jamais dans la couleur d'une série), identité = couleur ET forme.
import { h } from './dom.js';

const NS = 'http://www.w3.org/2000/svg';

export function s(tag, attrs = {}, ...children) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) if (value != null) node.setAttribute(key, value);
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function linear([d0, d1], [r0, r1]) {
  return (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
}

export function niceTicks(lo, hi, count = 5) {
  const raw = (hi - lo) / count;
  const mag = 10 ** Math.floor(Math.log10(raw || 1));
  const err = raw / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

// Domaine arrondi vers le haut : la dernière graduation couvre toujours les données.
export function niceMax(hi, count = 5) {
  const ticks = niceTicks(0, hi, count), step = ticks[1] - ticks[0];
  const max = Math.ceil(hi / step) * step;
  return { max, ticks: niceTicks(0, max, count) };
}

export const years = (t) => (Number.isInteger(t) ? String(t) : t.toLocaleString('fr-FR'));

export function widthOf(node, fallback = 720) {
  return Math.max(300, Math.round(node.getBoundingClientRect().width) || fallback);
}

// Cadre : renvoie le svg et la zone de tracé ; `title` donne un nom accessible au graphique.
export function frame(width, height, margin, title) {
  const svg = s('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', 'aria-label': title });
  return { svg, x0: margin.left, x1: width - margin.right, y0: height - margin.bottom, y1: margin.top };
}

export function axes(f, x, y, { xTicks, yTicks, xTitle, yTitle, fmtX = String, fmtY = String, grid = true }) {
  const g = s('g', { class: 'axis' });
  if (grid) g.append(s('g', { class: 'grid' }, yTicks.map((t) => s('line', { x1: f.x0, x2: f.x1, y1: y(t), y2: y(t) }))));
  g.append(...yTicks.map((t) => s('text', { x: f.x0 - 8, y: y(t) + 4, 'text-anchor': 'end' }, fmtY(t))));
  g.append(...xTicks.map((t) => s('text', { x: x(t), y: f.y0 + 18, 'text-anchor': 'middle' }, fmtX(t))));
  if (xTitle) g.append(s('text', { class: 'axis-title', x: f.x1, y: f.y0 + 38, 'text-anchor': 'end' }, xTitle));
  if (yTitle) g.append(s('text', { class: 'axis-title', x: f.x0 - 8, y: f.y1 - 12 }, yTitle));
  return g;
}

// États de l'examen : couleur ET forme (lisible sans la couleur, en noir et blanc et pour les daltoniens).
export const STATES = {
  untreated: { label: 'Sans traitement', color: 'var(--third)', shape: 'circle' },
  OFF: { label: 'OFF (traitement suspendu)', color: 'var(--up)', shape: 'square' },
  ON: { label: 'ON (sous traitement)', color: 'var(--down)', shape: 'triangle' },
};

export function marker(shape, cx, cy, r, color, hollow = false) {
  const paint = hollow ? { fill: 'var(--bg)', stroke: color, 'stroke-width': 2.5 }
    : { fill: color, class: 'mark-ring' };
  if (shape === 'square') return s('rect', { x: cx - r, y: cy - r, width: 2 * r, height: 2 * r, rx: 1.5, ...paint });
  if (shape === 'triangle') {
    const k = r * 1.25;
    return s('polygon', { points: `${cx},${cy - k} ${cx + k},${cy + k * 0.8} ${cx - k},${cy + k * 0.8}`, ...paint });
  }
  if (shape === 'diamond') return s('polygon', { points: `${cx},${cy - r * 1.3} ${cx + r * 1.3},${cy} ${cx},${cy + r * 1.3} ${cx - r * 1.3},${cy}`, ...paint });
  return s('circle', { cx, cy, r, ...paint });
}

// Légende : la clé reprend la marque (ligne pour une ligne, forme pour un point).
export function legend(items) {
  return h('ul', { class: 'legend' }, items.map(({ label, color, shape, line, dash, hollow }) => {
    const key = s('svg', { viewBox: '0 0 18 14', 'aria-hidden': 'true' },
      line ? s('line', { x1: 1, x2: 17, y1: 7, y2: 7, stroke: color, 'stroke-width': 2.5, 'stroke-dasharray': dash ? '4 3' : null })
        : marker(shape || 'circle', 9, 7, 4.5, color, hollow));
    return h('li', {}, key, label);
  }));
}

// Infobulle : la valeur d'abord, le libellé ensuite ; au survol ET au clavier ; zone de survol plus grande que la marque.
const tipNode = () => document.getElementById('tip');

function place(x, y) {
  const tip = tipNode();
  const { width, height } = tip.getBoundingClientRect();
  tip.style.left = `${Math.min(window.innerWidth - width - 8, x + 14)}px`;
  tip.style.top = `${Math.max(8, y - height - 12)}px`;
}

export function hideTip() { tipNode().hidden = true; }

export function bindTip(node, value, label) {
  node.setAttribute('tabindex', '0');
  node.setAttribute('aria-label', `${label} : ${value}`);
  const show = (x, y) => {
    const tip = tipNode();
    tip.replaceChildren(h('strong', {}, value), label);
    tip.hidden = false;
    place(x, y);
  };
  node.addEventListener('pointermove', (e) => show(e.clientX, e.clientY));
  node.addEventListener('pointerleave', hideTip);
  node.addEventListener('focus', () => { const r = node.getBoundingClientRect(); show(r.right, r.top); });
  node.addEventListener('blur', hideTip);
  return node;
}
