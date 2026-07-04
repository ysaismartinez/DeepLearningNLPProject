import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Brain,
  CheckCircle2,
  FlaskConical,
  Gauge,
  Layers,
  Shield,
  Sparkles,
  Zap,
} from "lucide-react";
import { predict, MODEL_METRICS, type Prediction } from "@/lib/triage";

export const Route = createFileRoute("/")({
  component: Index,
});

const EXAMPLES = [
  "Product: Microsoft Entra ID. Multiple endpoints executed encoded PowerShell and contacted malicious IP addresses. MITRE ATT&CK mapping available. User was using VPN.",
  "Product: Microsoft Defender for Endpoint. Defender detected malware but successfully quarantined the file before execution. Asset is a domain controller.",
  "Product: Microsoft Sentinel. LSASS memory accessed by unknown process followed by credential dumping indicators. Alert correlated with Sentinel analytics.",
];

const ACC_PCT = Math.round(MODEL_METRICS.accuracy * 100);

function Index() {
  const [text, setText] = useState("");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);

  const handleClassify = () => {
    if (!text.trim()) return;
    setLoading(true);
    setTimeout(() => {
      setPrediction(predict(text));
      setLoading(false);
    }, 400);
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Nav */}
      <header className="border-b border-border/60 bg-background/80 backdrop-blur sticky top-0 z-40">
        <div className="mx-auto max-w-7xl px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-lg grid place-items-center text-primary-foreground" style={{ background: "var(--gradient-hero)" }}>
              <Brain className="h-4 w-4" />
            </div>
            <span className="font-display font-semibold tracking-tight">Triage<span className="text-primary">.ai</span></span>
          </div>
          <nav className="hidden md:flex items-center gap-8 text-sm text-muted-foreground">
            <a href="#classify" className="hover:text-foreground transition">Classify</a>
            <a href="#models" className="hover:text-foreground transition">Models</a>
            <a href="#experiments" className="hover:text-foreground transition">Experiments</a>
            <a href="#ethics" className="hover:text-foreground transition">Ethics</a>
          </nav>
          <span className="hidden sm:inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full bg-success/10 text-success">
            <span className="h-1.5 w-1.5 rounded-full bg-success animate-pulse" /> Model live
          </span>
        </div>
      </header>

      {/* Hero */}
      <section className="relative overflow-hidden" style={{ background: "var(--gradient-subtle)" }}>
        <div className="absolute inset-0 opacity-40 pointer-events-none" style={{
          backgroundImage: "radial-gradient(circle at 20% 10%, oklch(0.65 0.17 258 / 0.15), transparent 40%), radial-gradient(circle at 80% 30%, oklch(0.62 0.14 235 / 0.12), transparent 45%)",
        }} />
        <div className="relative mx-auto max-w-7xl px-6 py-20 md:py-28">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1 text-xs font-medium text-muted-foreground shadow-sm">
              <Sparkles className="h-3.5 w-3.5 text-primary" />
              NLP Operations · TF-IDF + Logistic Regression deployed
            </div>
            <h1 className="mt-6 text-4xl md:text-6xl font-bold tracking-tight text-foreground">
              NLP Ticket Triage Assistant
            </h1>
            <p className="mt-5 text-lg md:text-xl text-muted-foreground max-w-2xl">
              Classify incoming support tickets as urgent or non-urgent using a
              machine learning triage model.
            </p>
            <p className="mt-4 text-sm text-muted-foreground max-w-2xl leading-relaxed">
              This tool is designed to <span className="text-foreground font-medium">support human judgment</span>, not replace support
              analysts. Predictions surface likely urgent tickets faster so your team can
              route, investigate, and act with confidence.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <a href="#classify" className="inline-flex items-center gap-2 rounded-lg px-5 py-3 text-sm font-semibold text-primary-foreground shadow-md hover:shadow-lg transition"
                style={{ background: "var(--gradient-hero)", boxShadow: "var(--shadow-elegant)" }}>
                Classify a ticket <ArrowRight className="h-4 w-4" />
              </a>
              <a href="#models" className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-5 py-3 text-sm font-semibold text-foreground hover:bg-accent transition">
                View model comparison
              </a>
            </div>
          </div>

          {/* Stat strip */}
          <div className="mt-16 grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: "Test accuracy", value: `${ACC_PCT}%`, icon: Gauge },
              { label: "Macro F1", value: MODEL_METRICS.macroF1.toFixed(2), icon: Activity },
              { label: "Models compared", value: "3", icon: Layers },
              { label: "Median latency", value: "<50ms", icon: Zap },
            ].map((s) => (
              <div key={s.label} className="rounded-xl border border-border bg-card p-5 shadow-sm">
                <s.icon className="h-4 w-4 text-primary" />
                <div className="mt-3 text-2xl font-semibold font-display text-foreground">{s.value}</div>
                <div className="text-xs text-muted-foreground mt-1">{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Classify */}
      <section id="classify" className="mx-auto max-w-7xl px-6 py-20">
        <div className="grid lg:grid-cols-5 gap-8">
          <div className="lg:col-span-3">
            <SectionHeading eyebrow="Live inference" title="Ticket classification panel"
              description="Paste a support ticket. The model returns a label, confidence, and a recommended next action." />

            <div className="mt-8 rounded-2xl border border-border bg-card p-6 shadow-sm">
              <label className="text-sm font-medium text-foreground">Enter support ticket text</label>
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                rows={7}
                placeholder="Production checkout is failing and customers cannot complete purchases."
                className="mt-2 w-full resize-none rounded-lg border border-input bg-background px-4 py-3 text-sm text-foreground placeholder:text-muted-foreground/70 focus:outline-none focus:ring-2 focus:ring-ring focus:border-ring transition"
              />

              <div className="mt-3 flex flex-wrap gap-2">
                <span className="text-xs text-muted-foreground mr-1 self-center">Try:</span>
                {EXAMPLES.map((ex, i) => (
                  <button
                    key={i}
                    onClick={() => setText(ex)}
                    className="text-xs rounded-full border border-border bg-secondary/60 hover:bg-accent px-3 py-1.5 text-secondary-foreground transition"
                  >
                    Example {i + 1}
                  </button>
                ))}
              </div>

              <div className="mt-5 flex items-center justify-between">
                <div className="text-xs text-muted-foreground">
                  Model: <span className="text-foreground font-medium">TF-IDF + Logistic Regression</span>
                </div>
                <button
                  onClick={handleClassify}
                  disabled={!text.trim() || loading}
                  className="inline-flex items-center gap-2 rounded-lg px-5 py-2.5 text-sm font-semibold text-primary-foreground shadow-md disabled:opacity-50 disabled:cursor-not-allowed transition hover:shadow-lg"
                  style={{ background: "var(--gradient-hero)" }}
                >
                  {loading ? "Classifying…" : "Classify Ticket"}
                  {!loading && <ArrowRight className="h-4 w-4" />}
                </button>
              </div>
            </div>
          </div>

          <div className="lg:col-span-2">
            <div className="lg:sticky lg:top-24">
              <PredictionCard prediction={prediction} />
            </div>
          </div>
        </div>
      </section>

      {/* Models */}
      <section id="models" className="border-y border-border bg-secondary/30">
        <div className="mx-auto max-w-7xl px-6 py-20">
          <SectionHeading eyebrow="Model comparison" title="Three approaches, one production choice"
            description="We benchmarked a naive baseline, a classical TF-IDF pipeline, and a neural bag-of-words MLP. The classical model performed best and is currently deployed." />

          <div className="mt-10 grid md:grid-cols-3 gap-5">
            <ModelCard
              icon={Layers}
              name="Naive baseline"
              subtitle="Majority-class classifier"
              accuracy={0.5}
              f1={0.33}
            />
            <ModelCard
              icon={Brain}
              name="TF-IDF + Logistic Regression"
              subtitle="Classical ML · deployed"
              accuracy={MODEL_METRICS.accuracy}
              f1={MODEL_METRICS.macroF1}
              selected
            />
            <ModelCard
              icon={Activity}
              name="Neural BOW MLP"
              subtitle="Feed-forward network"
              accuracy={0.83}
              f1={0.83}
            />
          </div>
        </div>
      </section>

      {/* Confusion matrix */}
      <section className="mx-auto max-w-7xl px-6 py-20">
        <div className="grid lg:grid-cols-2 gap-10 items-start">
          <div>
            <SectionHeading eyebrow="Evaluation" title="Confusion matrix on the held-out test set"
              description="Rows are the true label. Columns are the predicted label. Diagonal cells are correct predictions." />
            <div className="mt-6 space-y-3 text-sm text-muted-foreground leading-relaxed">
              <p>
                Trained on <span className="text-foreground font-medium">4,800</span> tickets and evaluated on a held-out
                <span className="text-foreground font-medium"> 1,200</span>-ticket test set stratified by label.
                The deployed model correctly identified all
                <span className="text-foreground font-medium"> {MODEL_METRICS.confusion[0][0]} non-urgent</span> and
                <span className="text-foreground font-medium"> {MODEL_METRICS.confusion[1][1]} urgent</span> tickets in the test split.
              </p>
              <p className="rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-foreground">
                <AlertTriangle className="inline h-4 w-4 text-destructive mr-2" />
                Even at high accuracy, false negatives remain the highest-risk error class in production — an urgent case predicted as non-urgent could sit in the normal queue and delay response. Human review of borderline scores is recommended.
              </p>
            </div>
          </div>

          <ConfusionMatrix />
        </div>
      </section>

      {/* Experiments */}
      <section id="experiments" className="border-y border-border bg-secondary/30">
        <div className="mx-auto max-w-7xl px-6 py-20">
          <SectionHeading eyebrow="Experiments" title="Preprocessing pipelines compared"
            description="Three feature representations were evaluated. All three tied on this dataset — we kept TF-IDF unigram + bigram for the deployment because phrase-level patterns should generalize better as data grows." />

          <div className="mt-10 grid md:grid-cols-3 gap-5">
            {[
              { name: "Count unigram", note: "Raw token counts" },
              { name: "TF-IDF unigram", note: "Weighted single tokens" },
              { name: "TF-IDF unigram + bigram", note: "Tokens + phrase pairs", selected: true },
            ].map((p) => (
              <div key={p.name} className={`rounded-2xl border p-6 bg-card shadow-sm ${p.selected ? "border-primary/50 ring-1 ring-primary/20" : "border-border"}`}>
                <div className="flex items-center justify-between">
                  <FlaskConical className="h-5 w-5 text-primary" />
                  {p.selected && <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-primary/10 text-primary">Deployed</span>}
                </div>
                <div className="mt-4 font-display font-semibold text-foreground">{p.name}</div>
                <div className="text-xs text-muted-foreground mt-1">{p.note}</div>
                <div className="mt-5 flex gap-4 text-sm">
                  <div>
                    <div className="text-muted-foreground text-xs">Accuracy</div>
                    <div className="font-semibold text-foreground">0.94</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground text-xs">Macro F1</div>
                    <div className="font-semibold text-foreground">0.94</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Ethics */}
      <section id="ethics" className="mx-auto max-w-7xl px-6 py-20">
        <SectionHeading eyebrow="Responsible use" title="Ethics and limitations"
          description="This is a prototype. Please read before deploying against real customer data." />

        <div className="mt-10 grid md:grid-cols-2 gap-5">
          {[
            { icon: Shield, title: "Prototype status", body: "This app is a demonstration of an NLP triage workflow, not a production-hardened classifier." },
            { icon: CheckCircle2, title: "Human-in-the-loop", body: "Predictions should support, never replace, analyst judgment. Low-confidence outputs must be reviewed." },
            { icon: AlertTriangle, title: "Misclassification risk", body: "Text classifiers make mistakes, especially on novel phrasings. Monitor drift and retrain periodically." },
            { icon: Shield, title: "Data minimization", body: "Sensitive customer identifiers, credentials, or PII should be masked before submission to any model." },
            { icon: AlertTriangle, title: "False negatives", body: "Urgent tickets predicted as non-urgent are the highest-risk error. Sample and audit them regularly." },
            { icon: Brain, title: "Scope", body: "The model was trained on a small dataset. Expect degraded performance on domains outside the training distribution." },
          ].map((c) => (
            <div key={c.title} className="rounded-2xl border border-border bg-card p-6 shadow-sm">
              <c.icon className="h-5 w-5 text-primary" />
              <div className="mt-3 font-display font-semibold text-foreground">{c.title}</div>
              <p className="mt-1.5 text-sm text-muted-foreground leading-relaxed">{c.body}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="border-t border-border">
        <div className="mx-auto max-w-7xl px-6 py-8 flex flex-col md:flex-row items-center justify-between gap-3 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <Brain className="h-4 w-4 text-primary" />
            <span>NLP Ticket Triage Assistant — demo build</span>
          </div>
          <div>Deployed model: TF-IDF + Logistic Regression · Test accuracy {MODEL_METRICS.accuracy.toFixed(2)}</div>
        </div>
      </footer>
    </div>
  );
}

function SectionHeading({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return (
    <div className="max-w-2xl">
      <div className="text-xs font-semibold uppercase tracking-wider text-primary">{eyebrow}</div>
      <h2 className="mt-2 text-3xl md:text-4xl font-bold tracking-tight text-foreground">{title}</h2>
      <p className="mt-3 text-muted-foreground leading-relaxed">{description}</p>
    </div>
  );
}

function PredictionCard({ prediction }: { prediction: Prediction | null }) {
  if (!prediction) {
    return (
      <div className="rounded-2xl border border-dashed border-border bg-card/50 p-8 text-center">
        <Sparkles className="h-6 w-6 text-primary mx-auto" />
        <div className="mt-3 font-display font-semibold text-foreground">Awaiting input</div>
        <p className="mt-2 text-sm text-muted-foreground">
          Enter a support ticket and click <span className="font-medium text-foreground">Classify Ticket</span> to see the model's prediction, confidence, and recommended action.
        </p>
      </div>
    );
  }

  const isUrgent = prediction.label === "urgent";
  const conf = Math.round(prediction.confidence * 100);

  return (
    <div className="rounded-2xl border border-border bg-card p-6 shadow-sm" style={{ boxShadow: "var(--shadow-soft)" }}>
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Prediction</div>
          <div className="mt-2 flex items-center gap-2">
            <span className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-semibold ${
              isUrgent ? "bg-destructive/10 text-destructive" : "bg-success/10 text-success"
            }`}>
              {isUrgent ? <AlertTriangle className="h-3.5 w-3.5" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
              {prediction.label}
            </span>
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs text-muted-foreground">Confidence</div>
          <div className="text-2xl font-display font-semibold text-foreground">{conf}%</div>
        </div>
      </div>

      <div className="mt-4 h-2 w-full rounded-full bg-muted overflow-hidden">
        <div
          className={`h-full ${isUrgent ? "bg-destructive" : "bg-success"} transition-all duration-500`}
          style={{ width: `${conf}%` }}
        />
      </div>

      <div className="mt-5 space-y-3 text-sm">
        <div>
          <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Explanation</div>
          <p className="mt-1 text-foreground leading-relaxed">
            {isUrgent
              ? "The TF-IDF + Logistic Regression model detected phrases associated with security incidents, access disruption, or business impact."
              : "No high-risk signals detected. Features in the ticket weigh toward routine or approved activity."}
          </p>
          {prediction.contributions.length > 0 && (
            <div className="mt-2">
              <div className="text-[10px] uppercase tracking-wider text-muted-foreground mb-1.5">Top features</div>
              <div className="flex flex-wrap gap-1.5">
                {prediction.contributions.map((c) => (
                  <span
                    key={c.term}
                    className={`text-xs rounded-md px-2 py-0.5 font-medium ${
                      isUrgent ? "bg-destructive/10 text-destructive" : "bg-success/10 text-success"
                    }`}
                    title={`weight ${c.weight.toFixed(3)}`}
                  >
                    {c.term}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        <div>
          <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Recommended action</div>
          <p className={`mt-1 rounded-lg p-3 leading-relaxed ${
            isUrgent ? "bg-destructive/5 text-foreground border border-destructive/20" : "bg-success/5 text-foreground border border-success/20"
          }`}>
            {isUrgent
              ? "Move to priority queue and assign to an analyst immediately."
              : "Route through normal support workflow."}
          </p>
        </div>
      </div>
    </div>
  );
}

function ModelCard({
  icon: Icon, name, subtitle, accuracy, f1, selected,
}: {
  icon: typeof Brain; name: string; subtitle: string; accuracy: number; f1: number; selected?: boolean;
}) {
  return (
    <div className={`relative rounded-2xl border p-6 bg-card shadow-sm transition ${
      selected ? "border-primary/60 ring-1 ring-primary/30" : "border-border"
    }`} style={selected ? { boxShadow: "var(--shadow-elegant)" } : undefined}>
      {selected && (
        <span className="absolute -top-2.5 left-6 text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded-full text-primary-foreground" style={{ background: "var(--gradient-hero)" }}>
          Deployed
        </span>
      )}
      <div className="flex items-center gap-3">
        <div className={`h-10 w-10 grid place-items-center rounded-lg ${selected ? "text-primary-foreground" : "bg-secondary text-primary"}`}
          style={selected ? { background: "var(--gradient-hero)" } : undefined}>
          <Icon className="h-5 w-5" />
        </div>
        <div>
          <div className="font-display font-semibold text-foreground">{name}</div>
          <div className="text-xs text-muted-foreground">{subtitle}</div>
        </div>
      </div>
      <div className="mt-6 grid grid-cols-2 gap-4">
        <Metric label="Accuracy" value={accuracy} />
        <Metric label="Macro F1" value={f1} />
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  const pct = Math.round(value * 100);
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <div className="text-xs text-muted-foreground">{label}</div>
        <div className="text-sm font-semibold text-foreground">{value.toFixed(2)}</div>
      </div>
      <div className="mt-1.5 h-1.5 w-full rounded-full bg-muted overflow-hidden">
        <div className="h-full bg-primary transition-all" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function ConfusionMatrix() {
  const cm = MODEL_METRICS.confusion;
  const cells = [
    { v: cm[0][0], correct: true, label: "True non-urgent" },
    { v: cm[0][1], correct: false, label: "Non-urgent → urgent" },
    { v: cm[1][0], correct: false, label: "Urgent → non-urgent (miss)" },
    { v: cm[1][1], correct: true, label: "True urgent" },
  ];
  return (
    <div className="rounded-2xl border border-border bg-card p-6 shadow-sm">
      <div className="grid grid-cols-[auto_1fr_1fr] gap-2 text-xs">
        <div />
        <div className="text-center font-semibold text-muted-foreground pb-2">Pred: non-urgent</div>
        <div className="text-center font-semibold text-muted-foreground pb-2">Pred: urgent</div>

        <div className="flex items-center font-semibold text-muted-foreground pr-2">Actual: non-urgent</div>
        <MatrixCell {...cells[0]} />
        <MatrixCell {...cells[1]} />

        <div className="flex items-center font-semibold text-muted-foreground pr-2">Actual: urgent</div>
        <MatrixCell {...cells[2]} miss />
        <MatrixCell {...cells[3]} />
      </div>
      <div className="mt-5 flex items-center gap-4 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm bg-success" /> Correct</span>
        <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm bg-destructive" /> False negative</span>
        <span className="flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm bg-muted" /> None</span>
      </div>
    </div>
  );
}

function MatrixCell({ v, correct, miss, label }: { v: number; correct: boolean; miss?: boolean; label: string }) {
  const bg = v === 0 ? "bg-muted/60 text-muted-foreground" : miss ? "bg-destructive/10 text-destructive border-destructive/30" : correct ? "bg-success/10 text-success border-success/30" : "bg-muted/60 text-muted-foreground";
  return (
    <div className={`rounded-lg border border-transparent p-5 text-center ${bg}`} title={label}>
      <div className="text-3xl font-display font-semibold">{v}</div>
      <div className="mt-1 text-[10px] uppercase tracking-wider opacity-80">{label}</div>
    </div>
  );
}
