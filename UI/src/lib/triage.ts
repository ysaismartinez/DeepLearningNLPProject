import model from "./triageModel.json";

// Mirrors sklearn TfidfVectorizer(ngram_range=(1,2), lowercase=True,
// token_pattern=r"(?u)\b\w\w+\b", norm='l2'), scored with LogisticRegression.
const TOKEN_RE = /[a-z0-9_]{2,}/g;

type Term = [number, number]; // [coef, idf]
const TERMS = model.terms as unknown as Record<string, Term>;
const INTERCEPT = model.intercept as number;

function tokenize(text: string): string[] {
  const lower = text.toLowerCase();
  const unigrams = lower.match(TOKEN_RE) ?? [];
  const grams: string[] = [...unigrams];
  for (let i = 0; i < unigrams.length - 1; i++) {
    grams.push(`${unigrams[i]} ${unigrams[i + 1]}`);
  }
  return grams;
}

export type Prediction = {
  label: "urgent" | "non_urgent";
  confidence: number;
  probability: number;
  contributions: { term: string; weight: number }[];
};

export function predict(text: string): Prediction | null {
  if (!text.trim()) return null;

  const tokens = tokenize(text);
  const counts = new Map<string, number>();
  for (const t of tokens) counts.set(t, (counts.get(t) ?? 0) + 1);

  // Build sparse TF-IDF vector, L2-normalize (matches sklearn default).
  const vec: { term: string; value: number; coef: number }[] = [];
  let norm = 0;
  for (const [term, tf] of counts) {
    const entry = TERMS[term];
    if (!entry) continue;
    const [coef, idf] = entry;
    const v = tf * idf;
    vec.push({ term, value: v, coef });
    norm += v * v;
  }
  norm = Math.sqrt(norm) || 1;

  let score = INTERCEPT;
  const contribs: { term: string; weight: number }[] = [];
  for (const { term, value, coef } of vec) {
    const contrib = (value / norm) * coef;
    score += contrib;
    contribs.push({ term, weight: contrib });
  }

  const p = 1 / (1 + Math.exp(-score));
  const label: "urgent" | "non_urgent" = p >= 0.5 ? "urgent" : "non_urgent";
  const confidence = label === "urgent" ? p : 1 - p;

  contribs.sort((a, b) =>
    label === "urgent" ? b.weight - a.weight : a.weight - b.weight,
  );

  return {
    label,
    confidence,
    probability: p,
    contributions: contribs.slice(0, 5).filter((c) =>
      label === "urgent" ? c.weight > 0 : c.weight < 0,
    ),
  };
}

export const MODEL_METRICS = model.metrics as {
  accuracy: number;
  macroF1: number;
  confusion: number[][];
};
