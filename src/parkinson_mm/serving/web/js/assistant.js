// Assistant documentaire : POST /rag/ask. Chaque réponse montre son statut (en mots ET par une forme), ses sources
// numérotées, ou un refus. Le texte de la réponse est inséré comme texte, jamais comme HTML.
import { api } from './api.js';
import { s } from './chart.js';
import { errorBox, h, pageHead } from './dom.js';

// Formulations reprises du jeu d'évaluation (q01, q13, q22) : l'assistant est sensible à la formulation (rag/README.md).
const SUGGESTIONS = [
  'Combien de patients atteints de Parkinson non traités la cohorte PPMI a-t-elle inclus au départ ?',
  'Dans ce projet, le GRU-D fait-il mieux que le MLP à 24 mois ?',
  'Quelle dose de lévodopa dois-je prendre le matin ?',
];
const STATUS = {
  generee: ['Réponse sourcée', 'circle'],
  extractif: ['Passages cités, sans rédaction', 'square'],
  citations_invalides: ['Réponse rédigée rejetée : citation absente, passages cités à la place', 'square'],
  introuvable: ['Introuvable dans le corpus', 'ring'],
  refus: ['Refusée : question médicale personnelle', 'cross'],
};
const thread = [];
let list = null, form = null;

function icon(kind) {
  const shape = {
    circle: s('circle', { cx: 7, cy: 7, r: 5, fill: 'var(--btn)' }),
    square: s('rect', { x: 2.5, y: 2.5, width: 9, height: 9, rx: 1.5, fill: 'var(--ref)' }),
    ring: s('circle', { cx: 7, cy: 7, r: 4.5, fill: 'none', stroke: 'var(--ref)', 'stroke-width': 2 }),
    cross: s('path', { d: 'M3 3l8 8M11 3l-8 8', stroke: 'var(--ink-2)', 'stroke-width': 2.4, 'stroke-linecap': 'round' }),
  }[kind];
  return s('svg', { viewBox: '0 0 14 14', 'aria-hidden': 'true' }, shape);
}

function answerBlock(item) {
  if (item.pending) return h('div', { class: 'a' }, h('p', { class: 'note' }, item.first
    ? 'Recherche dans le corpus… la première question charge le modèle d’embeddings (quelques secondes).'
    : 'Recherche dans le corpus…'));
  if (item.error) return errorBox(item.error);
  const r = item.result, [label, kind] = STATUS[r.status] ?? [r.status, 'square'];
  return h('div', { class: 'a' },
    h('p', { class: 'status' }, icon(kind), label),
    h('p', { class: 'answer' }, r.text),
    r.sources.length ? h('ol', { class: 'sources', 'aria-label': 'Sources' },
      r.sources.map((src) => h('li', {}, h('b', {}, `[${src.n}] `), `${src.citation}, ${src.where}`))) : null);
}

function render() {
  list.replaceChildren(...thread.slice().reverse().map((item) =>
    h('article', { class: 'qa' }, h('p', { class: 'q' }, item.question), answerBlock(item))));
}

async function ask(question) {
  const item = { question, pending: true, first: !thread.length };
  thread.push(item);
  render();
  try { item.result = await api('/rag/ask', { question }); } catch (error) { item.error = error.message; }
  item.pending = false;
  render();
}

export function show(root) {
  const input = h('textarea', { id: 'question', rows: 2, maxlength: 500, required: true, 'aria-label': 'Votre question',
    placeholder: 'Une question sur PPMI, les méthodes, les articles du corpus ou les résultats du projet' });
  form = h('form', { class: 'ask', onsubmit: (e) => {
    e.preventDefault();
    const question = input.value.trim();
    if (question.length >= 3) { input.value = ''; ask(question); }
  } }, input,
  h('button', { type: 'submit', class: 'btn' }, 'Poser la question'));
  list = h('div', { class: 'thread', 'aria-live': 'polite' });
  root.replaceChildren(
    pageHead('Assistant documentaire', 'Il répond à partir de quatre articles publiés et du README de ce projet, en citant '
      + 'chaque source. S’il ne trouve pas la réponse, il le dit ; il refuse les questions médicales personnelles.'),
    form,
    h('div', { class: 'suggest' }, SUGGESTIONS.map((q) => h('button', { type: 'button', onclick: () => ask(q) }, q))),
    list);
  render();
}
