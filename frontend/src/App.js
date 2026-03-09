import React, { useState, useCallback, useEffect } from "react";
import {
  RadialBarChart, RadialBar, PolarAngleAxis,
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from "recharts";

// ── API ───────────────────────────────────────────────────────────────────────
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

async function fetchPrediction(formData) {
  const res = await fetch(`${API_URL}/predict`, {
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

// ── Constants ─────────────────────────────────────────────────────────────────
const AGE_BRACKETS = [
  "[0-10)","[10-20)","[20-30)","[30-40)","[40-50)",
  "[50-60)","[60-70)","[70-80)","[80-90)","[90-100)",
];

const RISK_CONFIG = {
  "Low Risk":    { color: "#00d4aa", bg: "rgba(0,212,170,0.12)", label: "LOW",    icon: "●" },
  "Medium Risk": { color: "#f0a500", bg: "rgba(240,165,0,0.12)", label: "MEDIUM", icon: "◆" },
  "High Risk":   { color: "#ff4757", bg: "rgba(255,71,87,0.12)", label: "HIGH",   icon: "▲" },
};

const FEATURE_LABELS = {
  time_in_hospital:   "Days Hospitalised",
  num_lab_procedures: "Lab Procedures",
  num_procedures:     "Med. Procedures",
  num_medications:    "Medications",
  number_outpatient:  "Outpatient Visits",
  number_emergency:   "Emergency Visits",
  number_inpatient:   "Prior Inpatient Admits",
};

// ── Styles (CSS-in-JS) ────────────────────────────────────────────────────────
const styles = `
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg-void:    #060b18;
    --bg-deep:    #0a1020;
    --bg-panel:   #0e1628;
    --bg-card:    #131d35;
    --bg-input:   #1a2540;
    --border:     rgba(255,255,255,0.07);
    --border-hi:  rgba(100,160,255,0.25);
    --text-prime: #e8edf8;
    --text-mid:   #8898b8;
    --text-muted: #4a5a78;
    --accent:     #3d7eff;
    --accent-glow:#3d7eff55;
    --teal:       #00d4aa;
    --gold:       #f0a500;
    --red:        #ff4757;
    --font-serif: 'DM Serif Display', Georgia, serif;
    --font-body:  'Inter', system-ui, sans-serif;
    --font-mono:  'IBM Plex Mono', monospace;
  }

  html, body, #root {
    height: 100%;
    background: var(--bg-void);
    color: var(--text-prime);
    font-family: var(--font-body);
    font-size: 14px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg-deep); }
  ::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 3px; }

  /* Layout */
  .app {
    min-height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr;
  }

  /* Header */
  .header {
    background: var(--bg-deep);
    border-bottom: 1px solid var(--border);
    padding: 0 2rem;
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(12px);
  }
  .header-brand {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, var(--accent), #8b5cf6);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
  }
  .brand-name {
    font-family: var(--font-serif);
    font-size: 1.25rem;
    letter-spacing: -0.02em;
    color: var(--text-prime);
  }
  .brand-name span { color: var(--accent); }
  .header-status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-muted);
  }
  .status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 8px var(--teal);
    animation: pulse 2s infinite;
  }
  .status-dot.offline { background: var(--red); box-shadow: 0 0 8px var(--red); }
  @keyframes pulse {
    0%,100% { opacity: 1; } 50% { opacity: 0.4; }
  }

  /* Main layout */
  .main {
    display: grid;
    grid-template-columns: 420px 1fr;
    gap: 0;
    min-height: 0;
  }

  /* Sidebar */
  .sidebar {
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }

  /* Content area */
  .content {
    overflow-y: auto;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }

  /* Section headers */
  .section-title {
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
  }

  /* Card */
  .card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem;
    transition: border-color 0.2s;
  }
  .card:hover { border-color: var(--border-hi); }

  /* Form fields */
  .field {
    margin-bottom: 0.85rem;
  }
  .field label {
    display: block;
    font-size: 11px;
    font-weight: 500;
    color: var(--text-mid);
    margin-bottom: 5px;
    letter-spacing: 0.03em;
    text-transform: uppercase;
  }
  .field input, .field select {
    width: 100%;
    background: var(--bg-input);
    border: 1px solid var(--border);
    border-radius: 8px;
    color: var(--text-prime);
    font-family: var(--font-mono);
    font-size: 13px;
    padding: 9px 12px;
    outline: none;
    transition: border-color 0.2s, box-shadow 0.2s;
    appearance: none;
    -webkit-appearance: none;
  }
  .field input:focus, .field select:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-glow);
  }
  .field select option { background: var(--bg-card); }

  /* Field grid */
  .field-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0 0.75rem;
  }

  /* Submit button */
  .btn-predict {
    width: 100%;
    padding: 13px;
    border: none;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--accent), #6366f1);
    color: #fff;
    font-family: var(--font-mono);
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.08em;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s;
    box-shadow: 0 4px 20px rgba(61,126,255,0.3);
  }
  .btn-predict:hover:not(:disabled) {
    opacity: 0.9;
    transform: translateY(-1px);
    box-shadow: 0 8px 28px rgba(61,126,255,0.4);
  }
  .btn-predict:active:not(:disabled) { transform: translateY(0); }
  .btn-predict:disabled { opacity: 0.5; cursor: not-allowed; }

  /* Loading spinner */
  .spinner {
    width: 16px; height: 16px;
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    display: inline-block;
    vertical-align: middle;
    margin-right: 8px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Risk badge */
  .risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    border: 1px solid currentColor;
  }

  /* Probability gauge */
  .gauge-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    padding: 1.5rem 1rem;
  }
  .gauge-prob {
    font-family: var(--font-serif);
    font-size: 3.5rem;
    line-height: 1;
    letter-spacing: -0.02em;
  }
  .gauge-label {
    font-family: var(--font-mono);
    font-size: 10px;
    color: var(--text-muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }

  /* Stat grid */
  .stat-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1px;
    background: var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  .stat-cell {
    background: var(--bg-card);
    padding: 1rem;
    text-align: center;
  }
  .stat-val {
    font-family: var(--font-mono);
    font-size: 1.3rem;
    font-weight: 500;
    color: var(--accent);
  }
  .stat-key {
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 3px;
  }

  /* Clinical notes */
  .clinical-notes {
    background: rgba(61,126,255,0.05);
    border: 1px solid rgba(61,126,255,0.15);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 0.9rem 1rem;
    font-size: 12.5px;
    color: var(--text-mid);
    line-height: 1.7;
  }

  /* Feature chart label */
  .recharts-text { fill: var(--text-mid) !important; font-size: 11px; }

  /* Error alert */
  .alert-error {
    background: rgba(255,71,87,0.1);
    border: 1px solid rgba(255,71,87,0.3);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    color: #ff6b7a;
    font-size: 12px;
    display: flex;
    align-items: flex-start;
    gap: 8px;
  }

  /* Empty state */
  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    gap: 1rem;
    opacity: 0.5;
    padding: 3rem;
  }
  .empty-icon {
    font-size: 4rem;
    opacity: 0.3;
  }
  .empty-title {
    font-family: var(--font-serif);
    font-size: 1.4rem;
    color: var(--text-mid);
  }
  .empty-sub {
    font-size: 12px;
    color: var(--text-muted);
    max-width: 280px;
  }

  /* Confidence pill */
  .confidence-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-family: var(--font-mono);
    font-size: 10px;
    padding: 3px 10px;
    border-radius: 12px;
    background: rgba(255,255,255,0.05);
    color: var(--text-mid);
    border: 1px solid var(--border);
  }

  /* Row flex */
  .row { display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; }

  /* Animated entry */
  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .anim-entry { animation: fadeUp 0.4s ease both; }
  .anim-entry-d1 { animation-delay: 0.05s; }
  .anim-entry-d2 { animation-delay: 0.10s; }
  .anim-entry-d3 { animation-delay: 0.15s; }

  /* Responsive */
  @media (max-width: 900px) {
    .main { grid-template-columns: 1fr; }
    .sidebar { border-right: none; border-bottom: 1px solid var(--border); }
  }
`;

// ── Default form values ───────────────────────────────────────────────────────
const DEFAULTS = {
  age: "[70-80)",
  race: "Caucasian",
  gender: "Male",
  time_in_hospital: 5,
  num_lab_procedures: 40,
  num_procedures: 1,
  num_medications: 12,
  number_outpatient: 0,
  number_emergency: 1,
  number_inpatient: 2,
  diag_1: "250",
  diag_2: "401",
  diag_3: "428",
  insulin: "Up",
  change: "Ch",
  diabetesMed: "Yes",
};

// ── Custom Gauge tooltip ──────────────────────────────────────────────────────
const GaugeTooltip = () => null;

// ── Risk Gauge ────────────────────────────────────────────────────────────────
function RiskGauge({ probability, riskLevel }) {
  const cfg = RISK_CONFIG[riskLevel] || RISK_CONFIG["Low Risk"];
  const pct = Math.round(probability * 100);
  const data = [{ value: pct, fill: cfg.color }, { value: 100 - pct, fill: "transparent" }];

  return (
    <div className="gauge-container">
      <div style={{ position: "relative", width: 200, height: 120 }}>
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            cx="50%" cy="90%"
            innerRadius="70%" outerRadius="100%"
            startAngle={180} endAngle={0}
            data={data}
          >
            <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
            <RadialBar dataKey="value" cornerRadius={8} />
          </RadialBarChart>
        </ResponsiveContainer>
        <div style={{
          position: "absolute", bottom: 0, left: "50%", transform: "translateX(-50%)",
          textAlign: "center",
        }}>
          <div className="gauge-prob" style={{ color: cfg.color }}>
            {pct}%
          </div>
        </div>
      </div>
      <div className="gauge-label">Readmission Probability</div>
      <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
        <span className="risk-badge" style={{ color: cfg.color, borderColor: cfg.color + "55", background: cfg.bg }}>
          {cfg.icon} {riskLevel.toUpperCase()}
        </span>
      </div>
    </div>
  );
}

// ── Feature bar chart ─────────────────────────────────────────────────────────
function FeatureChart({ formData }) {
  const bars = Object.entries(FEATURE_LABELS).map(([k, label]) => ({
    name: label,
    value: Number(formData[k]) || 0,
  })).filter(d => d.value > 0);

  const colors = ["#3d7eff","#6366f1","#8b5cf6","#00d4aa","#f0a500","#ff4757","#ec4899"];

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={bars} layout="vertical" margin={{ left: 10, right: 20, top: 5, bottom: 5 }}>
        <XAxis type="number" tick={{ fill: "#4a5a78", fontSize: 10 }} axisLine={false} tickLine={false} />
        <YAxis dataKey="name" type="category" width={130} tick={{ fill: "#8898b8", fontSize: 11 }} axisLine={false} tickLine={false} />
        <Tooltip
          contentStyle={{ background: "#131d35", border: "1px solid rgba(255,255,255,0.07)", borderRadius: 8, fontSize: 12 }}
          labelStyle={{ color: "#e8edf8" }}
          itemStyle={{ color: "#8898b8" }}
          cursor={{ fill: "rgba(255,255,255,0.03)" }}
        />
        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
          {bars.map((_, i) => <Cell key={i} fill={colors[i % colors.length]} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

// ── Model info card ───────────────────────────────────────────────────────────
function ModelCard({ health }) {
  if (!health) return null;
  return (
    <div className="card anim-entry">
      <div className="section-title">System</div>
      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
          <span style={{ color: "var(--text-muted)" }}>Model</span>
          <span style={{ fontFamily: "var(--font-mono)", color: "var(--accent)" }}>
            {health.model_name || "—"}
          </span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
          <span style={{ color: "var(--text-muted)" }}>Status</span>
          <span style={{ color: health.model_loaded ? "var(--teal)" : "var(--red)", fontFamily: "var(--font-mono)" }}>
            {health.model_loaded ? "ONLINE" : "OFFLINE"}
          </span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
          <span style={{ color: "var(--text-muted)" }}>Uptime</span>
          <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-mid)" }}>
            {Math.floor((health.uptime_seconds || 0) / 60)}m {Math.round((health.uptime_seconds || 0) % 60)}s
          </span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
          <span style={{ color: "var(--text-muted)" }}>API</span>
          <span style={{ fontFamily: "var(--font-mono)", color: "var(--text-mid)" }}>v{health.api_version}</span>
        </div>
      </div>
    </div>
  );
}

// ── Field component ───────────────────────────────────────────────────────────
function Field({ label, name, type = "number", options, value, onChange, min, max }) {
  return (
    <div className="field">
      <label>{label}</label>
      {options ? (
        <select name={name} value={value} onChange={onChange}>
          {options.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
      ) : (
        <input
          type={type}
          name={name}
          value={value}
          min={min}
          max={max}
          onChange={onChange}
        />
      )}
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [form, setForm] = useState(DEFAULTS);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null);
  const [history, setHistory] = useState([]);

  // Fetch health on mount
  useEffect(() => {
    fetchHealth().then(h => setHealth(h)).catch(() => {});
  }, []);

  const handleChange = useCallback((e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  }, []);

  const handleSubmit = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      // Convert numeric fields
      const payload = {
        ...form,
        time_in_hospital:    Number(form.time_in_hospital),
        num_lab_procedures:  Number(form.num_lab_procedures),
        num_procedures:      Number(form.num_procedures),
        num_medications:     Number(form.num_medications),
        number_outpatient:   Number(form.number_outpatient),
        number_emergency:    Number(form.number_emergency),
        number_inpatient:    Number(form.number_inpatient),
      };
      const res = await fetchPrediction(payload);
      setResult(res);
      setHistory(prev => [{ ...res, timestamp: new Date().toLocaleTimeString(), age: form.age }, ...prev.slice(0, 4)]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [form]);

  const cfg = result ? (RISK_CONFIG[result.risk_level] || RISK_CONFIG["Low Risk"]) : null;

  return (
    <>
      <style>{styles}</style>
      <div className="app">

        {/* ── Header ── */}
        <header className="header">
          <div className="header-brand">
            <div className="brand-icon">⚕</div>
            <div>
              <div className="brand-name">Clinical<span>AI</span></div>
            </div>
          </div>
          <div style={{ flex: 1, textAlign: "center", display: "flex", justifyContent: "center" }}>
            <span style={{ fontFamily: "var(--font-serif)", fontSize: "1rem", color: "var(--text-mid)", letterSpacing: "0.05em" }}>
              Diabetes Readmission Risk · Decision Support System
            </span>
          </div>
          <div className="header-status">
            <div className={`status-dot ${health?.model_loaded === false ? "offline" : ""}`} />
            <span>{health ? (health.model_loaded ? "MODEL ONLINE" : "MODEL OFFLINE") : "CONNECTING…"}</span>
          </div>
        </header>

        {/* ── Main ── */}
        <div className="main">

          {/* ── Sidebar: Input Form ── */}
          <aside className="sidebar">
            <div>
              <div className="section-title">Patient Demographics</div>
              <div className="card">
                <Field label="Age Bracket" name="age" options={AGE_BRACKETS} value={form.age} onChange={handleChange} />
                <div className="field-grid">
                  <Field label="Race" name="race"
                    options={["Caucasian","AfricanAmerican","Hispanic","Asian","Other"]}
                    value={form.race} onChange={handleChange} />
                  <Field label="Gender" name="gender"
                    options={["Male","Female"]}
                    value={form.gender} onChange={handleChange} />
                </div>
              </div>
            </div>

            <div>
              <div className="section-title">Encounter Details</div>
              <div className="card">
                <div className="field-grid">
                  <Field label="Days in Hospital" name="time_in_hospital" min={1} max={14} value={form.time_in_hospital} onChange={handleChange} />
                  <Field label="Lab Procedures" name="num_lab_procedures" min={0} max={132} value={form.num_lab_procedures} onChange={handleChange} />
                  <Field label="Procedures" name="num_procedures" min={0} max={6} value={form.num_procedures} onChange={handleChange} />
                  <Field label="Medications" name="num_medications" min={0} max={81} value={form.num_medications} onChange={handleChange} />
                  <Field label="Outpatient Visits" name="number_outpatient" min={0} value={form.number_outpatient} onChange={handleChange} />
                  <Field label="Emergency Visits" name="number_emergency" min={0} value={form.number_emergency} onChange={handleChange} />
                  <Field label="Prior Inpatient" name="number_inpatient" min={0} value={form.number_inpatient} onChange={handleChange} />
                </div>
              </div>
            </div>

            <div>
              <div className="section-title">Diagnoses (ICD-9)</div>
              <div className="card">
                <div className="field-grid">
                  <Field label="Primary Dx" name="diag_1" type="text" value={form.diag_1} onChange={handleChange} />
                  <Field label="Secondary Dx" name="diag_2" type="text" value={form.diag_2} onChange={handleChange} />
                  <Field label="Tertiary Dx" name="diag_3" type="text" value={form.diag_3} onChange={handleChange} />
                </div>
              </div>
            </div>

            <div>
              <div className="section-title">Medication Management</div>
              <div className="card">
                <div className="field-grid">
                  <Field label="Insulin" name="insulin"
                    options={["No","Steady","Up","Down"]}
                    value={form.insulin} onChange={handleChange} />
                  <Field label="Med Change" name="change"
                    options={["Ch","No"]}
                    value={form.change} onChange={handleChange} />
                  <Field label="Diabetes Med" name="diabetesMed"
                    options={["Yes","No"]}
                    value={form.diabetesMed} onChange={handleChange} />
                </div>
              </div>
            </div>

            {error && (
              <div className="alert-error">
                <span>⚠</span>
                <span>{error}</span>
              </div>
            )}

            <button className="btn-predict" onClick={handleSubmit} disabled={loading}>
              {loading && <span className="spinner" />}
              {loading ? "ANALYSING PATIENT…" : "PREDICT READMISSION RISK"}
            </button>

            <ModelCard health={health} />
          </aside>

          {/* ── Content: Results ── */}
          <main className="content">
            {!result ? (
              <div className="empty-state">
                <div className="empty-icon">🏥</div>
                <div className="empty-title">No Prediction Yet</div>
                <div className="empty-sub">
                  Complete the patient intake form and click <strong>Predict Readmission Risk</strong> to generate a clinical decision support assessment.
                </div>
              </div>
            ) : (
              <>
                {/* Risk Score */}
                <div className="card anim-entry" style={{ borderColor: cfg.color + "33" }}>
                  <div className="section-title">Risk Assessment</div>
                  <RiskGauge probability={result.readmission_probability} riskLevel={result.risk_level} />
                  <div className="row" style={{ justifyContent: "center", marginTop: "0.5rem" }}>
                    <span className="confidence-pill">
                      ◎ CONFIDENCE: {result.confidence?.toUpperCase()}
                    </span>
                    <span className="confidence-pill">
                      ⚙ {result.model_name?.replace(/_/g, " ").toUpperCase()}
                    </span>
                  </div>
                </div>

                {/* Metrics */}
                <div className="card anim-entry anim-entry-d1">
                  <div className="section-title">Patient Metrics</div>
                  <div className="stat-grid">
                    <div className="stat-cell">
                      <div className="stat-val" style={{ color: "var(--teal)" }}>{form.time_in_hospital}d</div>
                      <div className="stat-key">LOS</div>
                    </div>
                    <div className="stat-cell">
                      <div className="stat-val" style={{ color: "var(--gold)" }}>{form.num_medications}</div>
                      <div className="stat-key">Meds</div>
                    </div>
                    <div className="stat-cell">
                      <div className="stat-val" style={{ color: form.number_emergency > 0 ? "var(--red)" : "var(--text-mid)" }}>
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
                      <div className="stat-val" style={{ color: form.diabetesMed === "Yes" ? "var(--accent)" : "var(--text-muted)" }}>
                        {form.diabetesMed}
                      </div>
                      <div className="stat-key">DM Med</div>
                    </div>
                  </div>
                </div>

                {/* Feature Chart */}
                <div className="card anim-entry anim-entry-d2">
                  <div className="section-title">Clinical Profile</div>
                  <FeatureChart formData={form} />
                </div>

                {/* Clinical Notes */}
                <div className="card anim-entry anim-entry-d3">
                  <div className="section-title">Clinical Decision Support Notes</div>
                  <div className="clinical-notes">
                    {result.clinical_notes}
                  </div>
                </div>

                {/* History */}
                {history.length > 1 && (
                  <div className="card anim-entry">
                    <div className="section-title">Recent Predictions</div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                      {history.map((h, i) => {
                        const hcfg = RISK_CONFIG[h.risk_level] || RISK_CONFIG["Low Risk"];
                        return (
                          <div key={i} style={{
                            display: "flex", justifyContent: "space-between", alignItems: "center",
                            padding: "8px 10px",
                            background: "var(--bg-input)",
                            borderRadius: 8,
                            fontSize: 12,
                          }}>
                            <span style={{ color: "var(--text-muted)", fontFamily: "var(--font-mono)" }}>{h.timestamp}</span>
                            <span style={{ color: "var(--text-mid)" }}>Age {h.age}</span>
                            <span style={{ fontFamily: "var(--font-mono)", color: hcfg.color }}>
                              {Math.round(h.readmission_probability * 100)}%
                            </span>
                            <span className="risk-badge" style={{ color: hcfg.color, borderColor: hcfg.color + "44", background: hcfg.bg, padding: "2px 10px" }}>
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
