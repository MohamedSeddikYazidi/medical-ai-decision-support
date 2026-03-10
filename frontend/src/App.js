import React, { useState, useCallback, useEffect } from "react";
import {
  RadialBarChart, RadialBar, PolarAngleAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

async function fetchPrediction(formData, modelName) {
  const url = modelName
    ? `${API_URL}/predict?model=${modelName}`
    : `${API_URL}/predict`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(formData),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Prediction failed");
  }
  return res.json();
}

async function fetchHealth() {
  const res = await fetch(`${API_URL}/health`);
  return res.ok ? res.json() : null;
}

async function fetchModelsList() {
  const res = await fetch(`${API_URL}/models/list`);
  return res.ok ? res.json() : null;
}

const AGE_BRACKETS = [
  "[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
  "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)",
];

const RISK_CONFIG = {
  "Low Risk":    { color: "#00d4aa", bg: "rgba(0,212,170,0.12)", icon: "●" },
  "Medium Risk": { color: "#f0a500", bg: "rgba(240,165,0,0.12)",  icon: "◆" },
  "High Risk":   { color: "#ff4757", bg: "rgba(255,71,87,0.12)",  icon: "▲" },
};

const MODEL_LABELS = {
  logistic_regression: "Logistic Regression",
  random_forest:       "Random Forest",
  xgboost:             "XGBoost",
  best_model:          "Best Model (auto)",
};

const MODEL_COLORS = {
  logistic_regression: "#3d7eff",
  random_forest:       "#00d4aa",
  xgboost:             "#f0a500",
};

const FEATURE_LABELS = {
  time_in_hospital:   "Days Hospitalised",
  num_lab_procedures: "Lab Procedures",
  num_procedures:     "Med. Procedures",
  num_medications:    "Medications",
  number_outpatient:  "Outpatient Visits",
  number_emergency:   "Emergency Visits",
  number_inpatient:   "Prior Inpatient",
};

const DEFAULTS = {
  age: "[70-80)", race: "Caucasian", gender: "Male",
  time_in_hospital: 5, num_lab_procedures: 40, num_procedures: 1,
  num_medications: 12, number_outpatient: 0, number_emergency: 1,
  number_inpatient: 2, diag_1: "250", diag_2: "401", diag_3: "428",
  insulin: "Up", change: "Ch", diabetesMed: "Yes",
};

// ── Styles ────────────────────────────────────────────────────────────────────
const styles = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg-void: #060b18; --bg-deep: #0a1020; --bg-panel: #0e1628;
    --bg-card: #131d35; --bg-input: #1a2540;
    --border: rgba(255,255,255,0.07); --border-hi: rgba(100,160,255,0.25);
    --text-prime: #e8edf8; --text-mid: #8898b8; --text-muted: #4a5a78;
    --accent: #3d7eff; --accent-glow: #3d7eff44;
    --teal: #00d4aa; --gold: #f0a500; --red: #ff4757;
    --font-serif: 'DM Serif Display', Georgia, serif;
    --font-body: 'Inter', system-ui, sans-serif;
    --font-mono: 'IBM Plex Mono', monospace;
  }
  html,body,#root { height:100%; background:var(--bg-void); color:var(--text-prime);
    font-family:var(--font-body); font-size:14px; -webkit-font-smoothing:antialiased; }
  ::-webkit-scrollbar{width:5px} ::-webkit-scrollbar-track{background:var(--bg-deep)}
  ::-webkit-scrollbar-thumb{background:var(--border-hi);border-radius:3px}
  .app { min-height:100vh; display:grid; grid-template-rows:auto 1fr; }
  .header { background:var(--bg-deep); border-bottom:1px solid var(--border);
    padding:0 1.5rem; height:60px; display:flex; align-items:center;
    justify-content:space-between; position:sticky; top:0; z-index:100; }
  .brand { display:flex; align-items:center; gap:10px; }
  .brand-icon { width:34px;height:34px;background:linear-gradient(135deg,var(--accent),#8b5cf6);
    border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:17px; }
  .brand-name { font-family:var(--font-serif);font-size:1.2rem;letter-spacing:-0.02em; }
  .brand-name span { color:var(--accent); }
  .header-status { display:flex;align-items:center;gap:8px;
    font-family:var(--font-mono);font-size:11px;color:var(--text-muted); }
  .dot { width:8px;height:8px;border-radius:50%;background:var(--teal);
    box-shadow:0 0 8px var(--teal);animation:pulse 2s infinite; }
  .dot.off { background:var(--red);box-shadow:0 0 8px var(--red); }
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  .main { display:grid; grid-template-columns:400px 1fr; min-height:0; }
  .sidebar { background:var(--bg-panel);border-right:1px solid var(--border);
    overflow-y:auto;padding:1.25rem;display:flex;flex-direction:column;gap:1rem; }
  .content { overflow-y:auto;padding:1.25rem;display:flex;flex-direction:column;gap:1rem; }
  .stitle { font-family:var(--font-mono);font-size:10px;font-weight:500;
    letter-spacing:.15em;text-transform:uppercase;color:var(--text-muted);
    margin-bottom:.65rem;display:flex;align-items:center;gap:8px; }
  .stitle::after { content:'';flex:1;height:1px;background:var(--border); }
  .card { background:var(--bg-card);border:1px solid var(--border);
    border-radius:12px;padding:1.1rem;transition:border-color .2s; }
  .card:hover { border-color:var(--border-hi); }
  .field { margin-bottom:.75rem; }
  .field label { display:block;font-size:10.5px;font-weight:500;color:var(--text-mid);
    margin-bottom:4px;letter-spacing:.03em;text-transform:uppercase; }
  .field input,.field select { width:100%;background:var(--bg-input);
    border:1px solid var(--border);border-radius:7px;color:var(--text-prime);
    font-family:var(--font-mono);font-size:13px;padding:8px 10px;outline:none;
    transition:border-color .2s,box-shadow .2s;appearance:none; }
  .field input:focus,.field select:focus {
    border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-glow); }
  .field select option { background:var(--bg-card); }
  .grid2 { display:grid;grid-template-columns:1fr 1fr;gap:0 .65rem; }

  /* Model selector */
  .model-selector { display:flex;flex-direction:column;gap:6px; }
  .model-btn { display:flex;align-items:center;justify-content:space-between;
    padding:9px 12px;background:var(--bg-input);border:1px solid var(--border);
    border-radius:8px;cursor:pointer;transition:all .15s;color:var(--text-mid);
    font-family:var(--font-mono);font-size:11px; }
  .model-btn:hover { border-color:var(--border-hi);color:var(--text-prime); }
  .model-btn.active { border-color:var(--accent);background:rgba(61,126,255,.08);
    color:var(--text-prime); }
  .model-btn .model-label { display:flex;align-items:center;gap:8px; }
  .model-dot { width:8px;height:8px;border-radius:50%; }
  .model-metrics { font-size:10px;color:var(--text-muted);text-align:right; }
  .best-badge { font-family:var(--font-mono);font-size:9px;padding:2px 7px;
    border-radius:10px;background:rgba(61,126,255,.15);color:var(--accent);
    border:1px solid rgba(61,126,255,.3); }

  .btn { width:100%;padding:12px;border:none;border-radius:9px;
    background:linear-gradient(135deg,var(--accent),#6366f1);color:#fff;
    font-family:var(--font-mono);font-size:12px;font-weight:500;
    letter-spacing:.08em;cursor:pointer;transition:opacity .2s,transform .15s;
    box-shadow:0 4px 18px rgba(61,126,255,.3); }
  .btn:hover:not(:disabled){opacity:.9;transform:translateY(-1px)}
  .btn:disabled{opacity:.5;cursor:not-allowed}
  .spinner{width:14px;height:14px;border:2px solid rgba(255,255,255,.3);
    border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;
    display:inline-block;vertical-align:middle;margin-right:6px}
  @keyframes spin{to{transform:rotate(360deg)}}

  .gauge-wrap { display:flex;flex-direction:column;align-items:center;gap:.5rem;padding:1.25rem .5rem; }
  .gauge-pct { font-family:var(--font-serif);font-size:3.2rem;line-height:1; }
  .gauge-lbl { font-family:var(--font-mono);font-size:10px;color:var(--text-muted);
    text-transform:uppercase;letter-spacing:.15em; }
  .risk-badge { display:inline-flex;align-items:center;gap:5px;padding:4px 12px;
    border-radius:18px;font-family:var(--font-mono);font-size:10.5px;font-weight:500;
    letter-spacing:.1em;border:1px solid currentColor; }

  .stat-grid { display:grid;grid-template-columns:repeat(3,1fr);gap:1px;
    background:var(--border);border-radius:9px;overflow:hidden; }
  .stat-cell { background:var(--bg-card);padding:.85rem;text-align:center; }
  .stat-val { font-family:var(--font-mono);font-size:1.2rem;font-weight:500;color:var(--accent); }
  .stat-key { font-size:9.5px;color:var(--text-muted);text-transform:uppercase;
    letter-spacing:.07em;margin-top:2px; }

  .notes { background:rgba(61,126,255,.05);border:1px solid rgba(61,126,255,.15);
    border-left:3px solid var(--accent);border-radius:7px;padding:.8rem .9rem;
    font-size:12.5px;color:var(--text-mid);line-height:1.7; }
  .alert { background:rgba(255,71,87,.1);border:1px solid rgba(255,71,87,.3);
    border-radius:7px;padding:.75rem .9rem;color:#ff6b7a;font-size:12px;
    display:flex;align-items:flex-start;gap:7px; }
  .empty { flex:1;display:flex;flex-direction:column;align-items:center;
    justify-content:center;text-align:center;gap:.85rem;opacity:.45;padding:3rem; }
  .empty-icon { font-size:3.5rem;opacity:.35; }
  .empty-title { font-family:var(--font-serif);font-size:1.35rem;color:var(--text-mid); }
  .empty-sub { font-size:12px;color:var(--text-muted);max-width:270px; }
  .pill { display:inline-flex;align-items:center;gap:4px;font-family:var(--font-mono);
    font-size:10px;padding:3px 9px;border-radius:11px;background:rgba(255,255,255,.05);
    color:var(--text-mid);border:1px solid var(--border); }
  .row { display:flex;align-items:center;gap:.65rem;flex-wrap:wrap; }
  @keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:translateY(0)}}
  .anim{animation:fadeUp .35s ease both}
  .d1{animation-delay:.05s} .d2{animation-delay:.1s} .d3{animation-delay:.15s}
  .history-row { display:flex;justify-content:space-between;align-items:center;
    padding:7px 9px;background:var(--bg-input);border-radius:7px;font-size:11.5px; }

  /* Comparison table */
  .compare-table { width:100%;border-collapse:collapse; }
  .compare-table th { font-family:var(--font-mono);font-size:10px;text-transform:uppercase;
    letter-spacing:.08em;color:var(--text-muted);padding:6px 10px;border-bottom:1px solid var(--border);
    text-align:left; }
  .compare-table td { padding:8px 10px;font-family:var(--font-mono);font-size:12px;
    color:var(--text-mid);border-bottom:1px solid rgba(255,255,255,.04); }
  .compare-table tr:last-child td { border-bottom:none; }
  .compare-table .best-row td { color:var(--text-prime);background:rgba(61,126,255,.04); }
  .compare-table .metric-hi { color:var(--teal); }
  @media(max-width:860px){.main{grid-template-columns:1fr}
    .sidebar{border-right:none;border-bottom:1px solid var(--border)}}
`;

// ── Sub-components ────────────────────────────────────────────────────────────
function Field({ label, name, type="number", options, value, onChange, min, max }) {
  return (
    <div className="field">
      <label>{label}</label>
      {options
        ? <select name={name} value={value} onChange={onChange}>
            {options.map(o => <option key={o} value={o}>{o}</option>)}
          </select>
        : <input type={type} name={name} value={value} min={min} max={max} onChange={onChange} />
      }
    </div>
  );
}

function ModelSelector({ models, selected, onSelect, bestModel }) {
  if (!models || models.length === 0) return null;
  return (
    <div>
      <div className="stitle">Select Model</div>
      <div className="card">
        <div className="model-selector">
          {/* "Best (auto)" option */}
          <div
            className={`model-btn ${!selected ? "active" : ""}`}
            onClick={() => onSelect(null)}
          >
            <div className="model-label">
              <div className="model-dot" style={{ background: "var(--accent)" }} />
              Best Model (auto)
            </div>
            <span className="best-badge">AUTO</span>
          </div>
          {models.map(m => (
            <div
              key={m.model_name}
              className={`model-btn ${selected === m.model_name ? "active" : ""}`}
              onClick={() => onSelect(m.model_name)}
            >
              <div className="model-label">
                <div className="model-dot"
                     style={{ background: MODEL_COLORS[m.model_name] || "var(--text-muted)" }} />
                {MODEL_LABELS[m.model_name] || m.model_name}
                {m.is_best && <span className="best-badge">BEST</span>}
              </div>
              <div className="model-metrics">
                AUC {m.val_roc_auc.toFixed(3)} · F1 {m.val_f1.toFixed(3)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function RiskGauge({ probability, riskLevel }) {
  const cfg = RISK_CONFIG[riskLevel] || RISK_CONFIG["Low Risk"];
  const pct = Math.round(probability * 100);
  const data = [
    { value: pct,       fill: cfg.color },
    { value: 100 - pct, fill: "transparent" },
  ];
  return (
    <div className="gauge-wrap">
      <div style={{ position: "relative", width: 190, height: 110 }}>
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart cx="50%" cy="90%" innerRadius="70%" outerRadius="100%"
                          startAngle={180} endAngle={0} data={data}>
            <PolarAngleAxis type="number" domain={[0,100]} tick={false} />
            <RadialBar dataKey="value" cornerRadius={7} />
          </RadialBarChart>
        </ResponsiveContainer>
        <div style={{ position:"absolute",bottom:0,left:"50%",transform:"translateX(-50%)",textAlign:"center" }}>
          <div className="gauge-pct" style={{ color: cfg.color }}>{pct}%</div>
        </div>
      </div>
      <div className="gauge-lbl">Readmission Probability</div>
      <span className="risk-badge"
            style={{ color: cfg.color, borderColor: cfg.color+"55", background: cfg.bg }}>
        {cfg.icon} {riskLevel.toUpperCase()}
      </span>
    </div>
  );
}

function FeatureChart({ formData }) {
  const colors = ["#3d7eff","#6366f1","#8b5cf6","#00d4aa","#f0a500","#ff4757","#ec4899"];
  const bars = Object.entries(FEATURE_LABELS)
    .map(([k, label]) => ({ name: label, value: Number(formData[k]) || 0 }))
    .filter(d => d.value > 0);
  return (
    <ResponsiveContainer width="100%" height={215}>
      <BarChart data={bars} layout="vertical" margin={{ left:8,right:18,top:4,bottom:4 }}>
        <XAxis type="number" tick={{ fill:"#4a5a78",fontSize:10 }} axisLine={false} tickLine={false} />
        <YAxis dataKey="name" type="category" width={125} tick={{ fill:"#8898b8",fontSize:11 }}
               axisLine={false} tickLine={false} />
        <Tooltip contentStyle={{ background:"#131d35",border:"1px solid rgba(255,255,255,.07)",
          borderRadius:7,fontSize:12 }} cursor={{ fill:"rgba(255,255,255,.03)" }} />
        <Bar dataKey="value" radius={[0,4,4,0]}>
          {bars.map((_,i) => <Cell key={i} fill={colors[i % colors.length]} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function ComparisonTable({ modelsData, currentModelName }) {
  if (!modelsData || modelsData.length === 0) return null;
  return (
    <div className="card anim d3">
      <div className="stitle">Model Comparison</div>
      <table className="compare-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Val AUC</th>
            <th>Val F1</th>
            <th>Test AUC</th>
          </tr>
        </thead>
        <tbody>
          {modelsData.map(m => (
            <tr key={m.model_name}
                className={m.is_best ? "best-row" : ""}>
              <td>
                <div style={{ display:"flex",alignItems:"center",gap:7 }}>
                  <div style={{ width:7,height:7,borderRadius:"50%",
                    background: MODEL_COLORS[m.model_name] || "var(--text-muted)" }} />
                  {MODEL_LABELS[m.model_name] || m.model_name}
                  {m.is_best && <span className="best-badge" style={{marginLeft:4}}>BEST</span>}
                  {m.model_name === currentModelName &&
                    <span className="best-badge" style={{marginLeft:4,borderColor:"var(--teal)",
                      color:"var(--teal)",background:"rgba(0,212,170,.1)"}}>ACTIVE</span>}
                </div>
              </td>
              <td className={m.is_best ? "metric-hi" : ""}>{m.val_roc_auc.toFixed(4)}</td>
              <td>{m.val_f1.toFixed(4)}</td>
              <td>{m.test_roc_auc.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [form, setForm]           = useState(DEFAULTS);
  const [result, setResult]       = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [health, setHealth]       = useState(null);
  const [modelsList, setModelsList] = useState([]);
  const [selectedModel, setSelectedModel] = useState(null); // null = auto/best
  const [history, setHistory]     = useState([]);

  useEffect(() => {
    fetchHealth().then(h => setHealth(h)).catch(() => {});
    fetchModelsList()
      .then(d => d && setModelsList(d.available || []))
      .catch(() => {});
  }, []);

  const handleChange = useCallback(e => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  }, []);

  const handleSubmit = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = {
        ...form,
        time_in_hospital:   Number(form.time_in_hospital),
        num_lab_procedures: Number(form.num_lab_procedures),
        num_procedures:     Number(form.num_procedures),
        num_medications:    Number(form.num_medications),
        number_outpatient:  Number(form.number_outpatient),
        number_emergency:   Number(form.number_emergency),
        number_inpatient:   Number(form.number_inpatient),
      };
      const res = await fetchPrediction(payload, selectedModel);
      setResult(res);
      setHistory(prev => [
        { ...res, ts: new Date().toLocaleTimeString(), age: form.age },
        ...prev.slice(0, 4),
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [form, selectedModel]);

  const cfg = result ? (RISK_CONFIG[result.risk_level] || RISK_CONFIG["Low Risk"]) : null;

  return (
    <>
      <style>{styles}</style>
      <div className="app">

        {/* Header */}
        <header className="header">
          <div className="brand">
            <div className="brand-icon">⚕</div>
            <div className="brand-name">Clinical<span>AI</span></div>
          </div>
          <span style={{ fontFamily:"var(--font-serif)",fontSize:".95rem",
            color:"var(--text-mid)",letterSpacing:".04em" }}>
            Diabetes Readmission Risk · Decision Support
          </span>
          <div className="header-status">
            <div className={`dot ${health?.model_loaded === false ? "off" : ""}`} />
            <span>{health ? (health.model_loaded ? "MODEL ONLINE" : "MODEL OFFLINE") : "CONNECTING…"}</span>
          </div>
        </header>

        <div className="main">
          {/* Sidebar */}
          <aside className="sidebar">

            {/* Model Selector */}
            <ModelSelector
              models={modelsList}
              selected={selectedModel}
              onSelect={setSelectedModel}
              bestModel={health?.model_name}
            />

            {/* Demographics */}
            <div>
              <div className="stitle">Demographics</div>
              <div className="card">
                <Field label="Age Bracket" name="age" options={AGE_BRACKETS} value={form.age} onChange={handleChange} />
                <div className="grid2">
                  <Field label="Race" name="race"
                    options={["Caucasian","AfricanAmerican","Hispanic","Asian","Other"]}
                    value={form.race} onChange={handleChange} />
                  <Field label="Gender" name="gender"
                    options={["Male","Female"]} value={form.gender} onChange={handleChange} />
                </div>
              </div>
            </div>

            {/* Encounter */}
            <div>
              <div className="stitle">Encounter Details</div>
              <div className="card">
                <div className="grid2">
                  <Field label="Days in Hospital" name="time_in_hospital" min={1} max={14} value={form.time_in_hospital} onChange={handleChange} />
                  <Field label="Lab Procedures"   name="num_lab_procedures" min={0} max={132} value={form.num_lab_procedures} onChange={handleChange} />
                  <Field label="Procedures"       name="num_procedures" min={0} max={6} value={form.num_procedures} onChange={handleChange} />
                  <Field label="Medications"      name="num_medications" min={0} max={81} value={form.num_medications} onChange={handleChange} />
                  <Field label="Outpatient Visits" name="number_outpatient" min={0} value={form.number_outpatient} onChange={handleChange} />
                  <Field label="Emergency Visits"  name="number_emergency" min={0} value={form.number_emergency} onChange={handleChange} />
                  <Field label="Prior Inpatient"   name="number_inpatient" min={0} value={form.number_inpatient} onChange={handleChange} />
                </div>
              </div>
            </div>

            {/* Diagnoses */}
            <div>
              <div className="stitle">Diagnoses (ICD-9)</div>
              <div className="card">
                <div className="grid2">
                  <Field label="Primary Dx"   name="diag_1" type="text" value={form.diag_1} onChange={handleChange} />
                  <Field label="Secondary Dx" name="diag_2" type="text" value={form.diag_2} onChange={handleChange} />
                  <Field label="Tertiary Dx"  name="diag_3" type="text" value={form.diag_3} onChange={handleChange} />
                </div>
              </div>
            </div>

            {/* Medications */}
            <div>
              <div className="stitle">Medications</div>
              <div className="card">
                <div className="grid2">
                  <Field label="Insulin"      name="insulin" options={["No","Steady","Up","Down"]} value={form.insulin} onChange={handleChange} />
                  <Field label="Med Change"   name="change"  options={["Ch","No"]}                  value={form.change}  onChange={handleChange} />
                  <Field label="Diabetes Med" name="diabetesMed" options={["Yes","No"]}             value={form.diabetesMed} onChange={handleChange} />
                </div>
              </div>
            </div>

            {error && <div className="alert"><span>⚠</span><span>{error}</span></div>}

            <button className="btn" onClick={handleSubmit} disabled={loading}>
              {loading && <span className="spinner" />}
              {loading
                ? "ANALYSING…"
                : `PREDICT${selectedModel ? " · " + (MODEL_LABELS[selectedModel] || selectedModel).toUpperCase() : ""}`}
            </button>
          </aside>

          {/* Content */}
          <main className="content">
            {!result ? (
              <div className="empty">
                <div className="empty-icon">🏥</div>
                <div className="empty-title">No Prediction Yet</div>
                <div className="empty-sub">
                  Choose a model, fill in the patient form, and click Predict to generate an assessment.
                </div>
              </div>
            ) : (
              <>
                {/* Gauge */}
                <div className="card anim" style={{ borderColor: cfg.color+"33" }}>
                  <div className="stitle">Risk Assessment</div>
                  <RiskGauge probability={result.readmission_probability} riskLevel={result.risk_level} />
                  <div className="row" style={{ justifyContent:"center", marginTop:".4rem" }}>
                    <span className="pill">◎ CONFIDENCE: {result.confidence?.toUpperCase()}</span>
                    <span className="pill">⚙ {(MODEL_LABELS[result.model_name] || result.model_name).toUpperCase()}</span>
                  </div>
                </div>

                {/* Stats */}
                <div className="card anim d1">
                  <div className="stitle">Patient Metrics</div>
                  <div className="stat-grid">
                    <div className="stat-cell">
                      <div className="stat-val" style={{color:"var(--teal)"}}>{form.time_in_hospital}d</div>
                      <div className="stat-key">LOS</div>
                    </div>
                    <div className="stat-cell">
                      <div className="stat-val" style={{color:"var(--gold)"}}>{form.num_medications}</div>
                      <div className="stat-key">Meds</div>
                    </div>
                    <div className="stat-cell">
                      <div className="stat-val" style={{color:form.number_emergency>0?"var(--red)":"var(--text-mid)"}}>
                        {form.number_emergency}
                      </div>
                      <div className="stat-key">ER Visits</div>
                    </div>
                    <div className="stat-cell">
                      <div className="stat-val">{form.num_lab_procedures}</div>
                      <div className="stat-key">Lab Tests</div>
                    </div>
                    <div className="stat-cell">
                      <div className="stat-val">{form.number_inpatient}</div>
                      <div className="stat-key">Prior Admits</div>
                    </div>
                    <div className="stat-cell">
                      <div className="stat-val" style={{color:form.diabetesMed==="Yes"?"var(--accent)":"var(--text-muted)"}}>
                        {form.diabetesMed}
                      </div>
                      <div className="stat-key">DM Med</div>
                    </div>
                  </div>
                </div>

                {/* Feature chart */}
                <div className="card anim d2">
                  <div className="stitle">Clinical Profile</div>
                  <FeatureChart formData={form} />
                </div>

                {/* Model comparison table */}
                <ComparisonTable
                  modelsData={modelsList}
                  currentModelName={result.model_name}
                />

                {/* Notes */}
                <div className="card anim d3">
                  <div className="stitle">Clinical Decision Support Notes</div>
                  <div className="notes">{result.clinical_notes}</div>
                </div>

                {/* History */}
                {history.length > 1 && (
                  <div className="card anim">
                    <div className="stitle">Recent Predictions</div>
                    <div style={{ display:"flex",flexDirection:"column",gap:5 }}>
                      {history.map((h, i) => {
                        const hcfg = RISK_CONFIG[h.risk_level] || RISK_CONFIG["Low Risk"];
                        return (
                          <div key={i} className="history-row">
                            <span style={{color:"var(--text-muted)",fontFamily:"var(--font-mono)"}}>{h.ts}</span>
                            <span style={{color:"var(--text-mid)"}}>{h.age}</span>
                            <span style={{fontFamily:"var(--font-mono)",color:"var(--text-mid)",fontSize:10}}>
                              {MODEL_LABELS[h.model_name] || h.model_name}
                            </span>
                            <span style={{fontFamily:"var(--font-mono)",color:hcfg.color}}>
                              {Math.round(h.readmission_probability*100)}%
                            </span>
                            <span className="risk-badge"
                                  style={{color:hcfg.color,borderColor:hcfg.color+"44",background:hcfg.bg,padding:"2px 9px"}}>
                              {h.risk_level}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </>
            )}
          </main>
        </div>
      </div>
    </>
  );
}