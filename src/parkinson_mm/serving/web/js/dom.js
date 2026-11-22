// Petits outils DOM. Tout texte venant de l'API passe par des nœuds texte (jamais innerHTML) :
// noms de facteurs, réponses de l'assistant et sources sont des données, pas du HTML.

export function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  setAttrs(node, attrs);
  append(node, children);
  return node;
}

export function setAttrs(node, attrs) {
  for (const [key, value] of Object.entries(attrs)) {
    if (value == null || value === false) continue;
    if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (key === 'class') node.setAttribute('class', value);
    else node.setAttribute(key, value === true ? '' : value);
  }
}

export function append(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export const fr = (x, digits = 1) =>          // signe moins typographique (−), pas le tiret court
  Number(x).toLocaleString('fr-FR', { minimumFractionDigits: digits, maximumFractionDigits: digits }).replace('-', '−');

export const signed = (x, digits = 1) => (x > 0 ? '+' : x < 0 ? '−' : '') + fr(Math.abs(x), digits);

export const interval = (lo, hi, digits = 1) => `[${fr(lo, digits)} ; ${fr(hi, digits)}]`;

export function debounce(fn, ms = 350) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}

export function errorBox(message) {
  return h('p', { class: 'error', role: 'alert' }, message);
}

export function pageHead(title, lede) {
  return h('header', { class: 'head' }, h('h1', {}, title), h('p', { class: 'lede' }, lede));
}

export function store(key, fallback) {
  try { return JSON.parse(sessionStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}

export function save(key, value) {
  try { sessionStorage.setItem(key, JSON.stringify(value)); } catch { /* navigation privée : rien à garder */ }
}
