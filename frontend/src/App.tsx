// VERITAS Media Forensics — App.tsx
// This file preserves the existing investigation, forensic analysis,
// history, reports, modules, authentication, and system functionality.

import {
  useEffect,
  useMemo,
  useState,
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
  type ReactNode,
} from "react";
import "./App.css";
import {
  authenticateWithPasskey,
  registerPasskey,
} from "./webauthn";

type Page =
  | "dashboard"
  | "investigation"
  | "history"
  | "reports"
  | "modules"
  | "system";

type AnalysisResult = {
  filename?: string;
  content_type?: string;
  image_width?: number;
  image_height?: number;
  faces_detected?: number;
  prediction?: string;
  verdict?: string;
  confidence?: number;
  real_probability?: number;
  fake_probability?: number;
  model_name?: string;
  model_version?: string;
  explainability_method?: string;
  explainability_status?: string;
  explainability?: {
    method?: string;
    status?: string;
    heatmap_base64?: string | null;
  };
  processing_time_ms?: number;
  report?: {
    available?: boolean;
    filename?: string;
  };
  trained_for_deepfake_detection?: boolean;
  analysis_id?: number;
  image?: {
    width?: number;
    height?: number;
  };
  model?: {
    name?: string;
    prediction?: string;
    confidence?: number;
    real_probability?: number;
    fake_probability?: number;
    trained_for_deepfake_detection?: boolean;
    class_names?: string[];
    trained_epoch?: number | null;
    validation_accuracy?: number | null;
  };
  face_detection?: {
    faces_detected?: number;
    count?: number;
    method?: string;
    selected_face?: {
      x?: number;
      y?: number;
      width?: number;
      height?: number;
    } | null;
  };
  metadata?: {
    format?: string;
    exif_available?: boolean;
    camera_make?: string | null;
    camera_model?: string | null;
    date_time?: string | null;
    software?: string | null;
    orientation?: string | number | null;
    gps_present?: boolean;
    fields?: Record<string, string | number | boolean | null>;
  };
  image_integrity?: {
    status?: string;
    format?: string;
    file_size_bytes?: number;
    width?: number;
    height?: number;
    color_mode?: string;
    has_alpha_channel?: boolean;
    has_exif?: boolean;
    exif_field_count?: number;
    image_verified?: boolean;
    dimensions_valid?: boolean;
    file_size_valid?: boolean;
    sha256?: string;
    warnings?: string[];
    forensic_note?: string;
  };
  compression_analysis?: {
    status?: string;
    format?: string;
    file_size_bytes?: number;
    width?: number;
    height?: number;
    pixel_count?: number;
    bytes_per_pixel?: number;
    estimated_compression_level?: string;
    jpeg_quality_information?: string | null;
    jpeg_progressive?: boolean | null;
    compression_check_passed?: boolean;
    warnings?: string[];
    forensic_note?: string;
  };
  noise_frequency_analysis?: {
    status?: string;
    noise_level?: string;
    noise_variance?: number;
    noise_standard_deviation?: number;
    sharpness_variance?: number;
    frequency_energy?: number;
    high_frequency_energy_ratio?: number;
    frequency_characteristic?: string;
    low_frequency_base64?: string | null;
    high_frequency_base64?: string | null;
    noise_map_base64?: string | null;
    fft_spectrum_base64?: string | null;
    noise_heatmap_base64?: string | null;
    warnings?: string[];
    forensic_note?: string;
  };
  error?: string;
  message?: string;
};

type HistoryItem = {
  id: number;
  filename: string;
  verdict: string;
  confidence: number;
  faces: number;
  processingTime: number;
  timestamp: string;
};

type ModuleInfo = {
  number: string;
  title: string;
  description: string;
  status: "ACTIVE" | "PLANNED";
  icon: string;
};

const API_URL = "http://127.0.0.1:8000";

const AUTH_USERNAME = "analyst";
const AUTH_PASSWORD = "VERITAS@2026";
const AUTH_SESSION_KEY = "veritas_authenticated";

const modules: ModuleInfo[] = [
  {
    number: "01",
    title: "Face Detection",
    description:
      "Detects facial regions in submitted media using the forensic vision pipeline.",
    status: "ACTIVE",
    icon: "◉",
  },
  {
    number: "02",
    title: "Deepfake Classification",
    description:
      "EfficientNet-B0 based classification engine for real/fake media analysis.",
    status: "ACTIVE",
    icon: "◆",
  },
  {
    number: "03",
    title: "Confidence Analysis",
    description:
      "Reports the model confidence and class probability distribution.",
    status: "ACTIVE",
    icon: "◈",
  },
  {
    number: "04",
    title: "Grad-CAM Explainability",
    description:
      "Visual explanation layer highlighting regions influencing the model.",
    status: "ACTIVE",
    icon: "✦",
  },
  {
    number: "05",
    title: "Metadata Forensics",
    description:
      "EXIF and embedded metadata inspection for image provenance analysis.",
    status: "ACTIVE",
    icon: "▣",
  },
  {
    number: "06",
    title: "Compression Analysis",
    description:
      "JPEG compression and image integrity indicators for forensic review.",
    status: "ACTIVE",
    icon: "▤",
  },
  {
    number: "07",
    title: "Frequency Analysis",
    description:
      "Noise and frequency-domain indicators returned by the forensic engine.",
    status: "ACTIVE",
    icon: "⌁",
  },
  {
    number: "08",
    title: "Forensic Reporting",
    description:
      "Generated forensic reports and evidence summaries for completed analyses.",
    status: "ACTIVE",
    icon: "▥",
  },
  {
    number: "09",
    title: "Image Integrity Analysis",
    description:
      "Verifies image structure, dimensions, file properties, EXIF presence, and SHA-256 evidence integrity signals.",
    status: "ACTIVE",
    icon: "✓",
  },
  {
    number: "10",
    title: "Multi-Face Analysis",
    description:
      "Analyzes every detected face and consolidates face-level probabilities into an image-level forensic result.",
    status: "ACTIVE",
    icon: "◎",
  },
];

function formatSafeNumber(
  value: number | null | undefined,
  decimals = 2
): string {
  if (value == null || !Number.isFinite(Number(value))) {
    return "—";
  }
  return Number(value).toFixed(decimals);
}

function formatPercentValue(
  value: number | null | undefined
): string {
  if (value == null || !Number.isFinite(Number(value))) {
    return "—";
  }

  const number = Number(value);

  return number <= 1
    ? `${formatSafeNumber(number * 100, 2)}%`
    : `${formatSafeNumber(number, 2)}%`;
}

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(() => {
    return sessionStorage.getItem(AUTH_SESSION_KEY) === "true";
  });

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [loginError, setLoginError] = useState("");

  function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoginError("");

    const enteredUsername = username.trim();

    if (!enteredUsername || !password) {
      setLoginError(
        "Please enter both your analyst username and password."
      );
      return;
    }

    if (
      enteredUsername === AUTH_USERNAME &&
      password === AUTH_PASSWORD
    ) {
      sessionStorage.setItem(AUTH_SESSION_KEY, "true");
      setIsAuthenticated(true);
      setUsername("");
      setPassword("");
      setLoginError("");
      return;
    }

    setLoginError(
      "Authentication failed. Check your username and password."
    );
  }

  async function handlePasskeyLogin() {
    setLoginError("");

    try {
      const result = await authenticateWithPasskey();

      if (result.success) {
        sessionStorage.setItem(AUTH_SESSION_KEY, "true");
        setIsAuthenticated(true);
        setUsername("");
        setPassword("");
        setLoginError("");
        return;
      }

      setLoginError(result.message);
    } catch (error) {
      setLoginError(
        error instanceof Error
          ? error.message
          : "Passkey authentication failed."
      );
    }
  }

  function handleLogout() {
    sessionStorage.removeItem(AUTH_SESSION_KEY);
    setIsAuthenticated(false);
    setUsername("");
    setPassword("");
    setLoginError("");
  }

  if (!isAuthenticated) {
    return (
      <LoginPage
        username={username}
        password={password}
        showPassword={showPassword}
        loginError={loginError}
        setUsername={setUsername}
        setPassword={setPassword}
        setShowPassword={setShowPassword}
        onLogin={handleLogin}
        onPasskeyLogin={handlePasskeyLogin}
      />
    );
  }

  return <AuthenticatedApp onLogout={handleLogout} />;
}

function LoginPage({
  username,
  password,
  showPassword,
  loginError,
  setUsername,
  setPassword,
  setShowPassword,
  onLogin,
  onPasskeyLogin,
}: {
  username: string;
  password: string;
  showPassword: boolean;
  loginError: string;
  setUsername: (value: string) => void;
  setPassword: (value: string) => void;
  setShowPassword: (value: boolean) => void;
  onLogin: (event: FormEvent<HTMLFormElement>) => void;
  onPasskeyLogin: () => Promise<void>;
}) {
  async function handlePasskeyRegistration() {
    try {
      const result = await registerPasskey();
      alert(result.message);
    } catch (error) {
      alert(
        error instanceof Error
          ? error.message
          : "Passkey registration failed."
      );
    }
  }

  return (
    <div className="login-screen">
      <div className="login-background-grid" />

      <div className="login-card">
        <div className="login-brand">
          <div className="login-brand-symbol">V</div>

          <div>
            <div className="login-brand-name">VERITAS</div>
            <div className="login-brand-subtitle">MEDIA FORENSICS</div>
          </div>
        </div>

        <div className="login-header">
          <div className="login-eyebrow">SECURE ANALYST ACCESS</div>

          <h1>Forensic Workspace</h1>

          <p>
            Authenticate to access the VERITAS digital media forensic
            environment.
          </p>
        </div>

        <form className="login-form" onSubmit={onLogin}>
          <div className="login-field">
            <label htmlFor="veritas-username">ANALYST USERNAME</label>

            <div className="login-input-wrapper">
              <span className="login-input-icon">◉</span>

              <input
                id="veritas-username"
                type="text"
                value={username}
                onChange={(event) => setUsername(event.target.value)}
                placeholder="Enter analyst username"
                autoComplete="username"
                autoFocus
              />
            </div>
          </div>

          <div className="login-field">
            <label htmlFor="veritas-password">PASSWORD</label>

            <div className="login-input-wrapper">
              <span className="login-input-icon">◆</span>

              <input
                id="veritas-password"
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="Enter secure password"
                autoComplete="current-password"
              />

              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowPassword(!showPassword)}
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? "◉" : "◌"}
              </button>
            </div>
          </div>

          {loginError && (
            <div className="login-error">
              <span className="login-error-icon">!</span>
              <span>{loginError}</span>
            </div>
          )}

          <button type="submit" className="login-button">
            <span>Authenticate</span>
            <span className="login-button-arrow">→</span>
          </button>

          <button
            type="button"
            className="biometric-button"
            onClick={handlePasskeyRegistration}
          >
            <span>◉</span>
            <span>Register Passkey</span>
          </button>

          <button
            type="button"
            className="biometric-button"
            onClick={onPasskeyLogin}
          >
            <span>◉</span>
            <span>Login with Passkey</span>
          </button>
        </form>

        <div className="login-security">
          <div className="login-security-icon">✓</div>

          <div>
            <strong>SECURE LOCAL SESSION</strong>
            <span>
              Authentication session is maintained for this browser session.
            </span>
          </div>
        </div>

        <div className="login-divider">
          <span />
          <span>VERITAS</span>
          <span />
        </div>

        <div className="login-footer">
          <span>DIGITAL MEDIA FORENSICS</span>
          <span>ENGINE v1.0</span>
        </div>
      </div>
    </div>
  );
}

function AuthenticatedApp({ onLogout }: { onLogout: () => void }) {
  const [page, setPage] = useState<Page>("dashboard");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [backendOnline, setBackendOnline] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  const activeModuleCount = useMemo(
    () => modules.filter((module) => module.status === "ACTIVE").length,
    []
  );

  const plannedModuleCount = useMemo(
    () => modules.filter((module) => module.status === "PLANNED").length,
    []
  );

  async function loadHistory() {
    try {
      const response = await fetch(`${API_URL}/history`);

      if (!response.ok) {
        throw new Error("Failed to load analysis history.");
      }

      const data = await response.json();

      const records: HistoryItem[] = (data.records || []).map(
        (record: any) => ({
          id: Number(record.id),
          filename: record.filename || "Unknown evidence",
          verdict: String(record.prediction || "UNKNOWN").toUpperCase(),
          confidence: Number(record.confidence ?? 0),
          faces: Number(record.faces_detected ?? 0),
          processingTime: Number(record.processing_time_ms ?? 0),
          timestamp: record.created_at
            ? new Date(record.created_at).toLocaleString()
            : "—",
        })
      );

      setHistory(records);
    } catch (error) {
      console.error("History loading error:", error);
    }
  }

  useEffect(() => {
    void loadHistory();
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  async function checkBackend() {
    try {
      const response = await fetch(`${API_URL}/health`);

      if (!response.ok) {
        setBackendOnline(false);
        return;
      }

      const data = await response.json();
      setBackendOnline(data.status === "healthy");
    } catch {
      setBackendOnline(false);
    }
  }

  async function analyzeFile(file: File) {
    if (!file) {
      return;
    }

    const allowedTypes = ["image/jpeg", "image/png", "image/webp"];

    if (!allowedTypes.includes(file.type)) {
      window.alert(
        "Unsupported file type.\n\nPlease upload a JPEG, PNG, or WebP image."
      );
      return;
    }

    setSelectedFile(file);
    setResult(null);

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    const newPreviewUrl = URL.createObjectURL(file);
    setPreviewUrl(newPreviewUrl);
    setIsAnalyzing(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      const data: AnalysisResult = await response.json();

      if (!response.ok) {
        throw new Error(
          data.message ||
            data.error ||
            "The forensic analysis request failed."
        );
      }

      setResult(data);
      setBackendOnline(true);

      const verdict = String(
        data.prediction || data.verdict || "UNKNOWN"
      ).toUpperCase();

      const confidence = Number(data.confidence || 0);

      const historyItem: HistoryItem = {
        id: Number(data.analysis_id ?? Date.now()),
        filename: data.filename || file.name,
        verdict,
        confidence,
        faces: Number(data.faces_detected || 0),
        processingTime: Number(data.processing_time_ms || 0),
        timestamp: new Date().toLocaleString(),
      };

      setHistory((previous) => [
        historyItem,
        ...previous.filter((item) => item.id !== historyItem.id),
      ]);

      void loadHistory();
    } catch (error) {
      setBackendOnline(false);

      const message =
        error instanceof Error
          ? error.message
          : "Unable to connect to the forensic analysis service.";

      setResult({ error: message });
    } finally {
      setIsAnalyzing(false);
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];

    if (file) {
      void analyzeFile(file);
    }

    event.target.value = "";
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);

    const file = event.dataTransfer.files?.[0];

    if (file) {
      void analyzeFile(file);
    }
  }

  function handleDragOver(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(true);
  }

  function handleDragLeave(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
  }

  function clearInvestigation() {
    setSelectedFile(null);
    setResult(null);

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    setPreviewUrl("");
  }

  function navigate(nextPage: Page) {
    setPage(nextPage);
    setSidebarOpen(false);
  }

  function formatPercent(value?: number | null) {
    return formatPercentValue(value);
  }

  function getConfidencePercent(value?: number | null) {
    if (value == null || !Number.isFinite(Number(value))) {
      return 0;
    }

    const number = Number(value);
    return number <= 1 ? number * 100 : number;
  }

  function getVerdict() {
    if (!result) {
      return "WAITING";
    }

    if (result.error) {
      return "ERROR";
    }

    return String(
      result.model?.prediction ||
        result.prediction ||
        result.verdict ||
        "UNKNOWN"
    ).toUpperCase();
  }

  function renderVerdictClass() {
    const verdict = getVerdict();

    if (verdict === "FAKE") {
      return "verdict-fake";
    }

    if (verdict === "REAL") {
      return "verdict-real";
    }

    if (verdict === "ERROR") {
      return "verdict-error";
    }

    return "verdict-neutral";
  }

  return (
    <div className="app-shell">
      <aside
        className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}
      >
        <div className="brand-area">
          <div className="brand-mark">V</div>

          <div>
            <div className="brand-name">VERITAS</div>
            <div className="brand-subtitle">MEDIA FORENSICS</div>
          </div>
        </div>

        <div className="system-indicator">
          <span
            className={`status-dot ${
              backendOnline ? "status-online" : "status-offline"
            }`}
          />
          <span>
            {backendOnline
              ? "FORENSIC ENGINE ONLINE"
              : "ENGINE STATUS UNKNOWN"}
          </span>
        </div>

        <nav className="sidebar-navigation">
          <div className="nav-label">INVESTIGATION</div>

          <button
            className={`nav-item ${
              page === "dashboard" ? "nav-item-active" : ""
            }`}
            onClick={() => navigate("dashboard")}
          >
            <span className="nav-icon">⌂</span>
            <span>Command Center</span>
          </button>

          <button
            className={`nav-item ${
              page === "investigation" ? "nav-item-active" : ""
            }`}
            onClick={() => navigate("investigation")}
          >
            <span className="nav-icon">⊕</span>
            <span>New Investigation</span>
          </button>

          <button
            className={`nav-item ${
              page === "history" ? "nav-item-active" : ""
            }`}
            onClick={() => navigate("history")}
          >
            <span className="nav-icon">◷</span>
            <span>Investigation History</span>
          </button>

          <div className="nav-label">FORENSICS</div>

          <button
            className={`nav-item ${
              page === "modules" ? "nav-item-active" : ""
            }`}
            onClick={() => navigate("modules")}
          >
            <span className="nav-icon">◈</span>
            <span>Forensic Modules</span>
          </button>

          <button
            className={`nav-item ${
              page === "reports" ? "nav-item-active" : ""
            }`}
            onClick={() => navigate("reports")}
          >
            <span className="nav-icon">▤</span>
            <span>Forensic Reports</span>
          </button>

          <div className="nav-label">EVIDENCE</div>

          <button
            className="nav-item"
            onClick={() => navigate("investigation")}
          >
            <span className="nav-icon">◉</span>
            <span>Evidence Workspace</span>
          </button>

          <button
            className="nav-item"
            onClick={() => navigate("history")}
          >
            <span className="nav-icon">▣</span>
            <span>Case Records</span>
          </button>

          <div className="nav-label">SYSTEM</div>

          <button
            className={`nav-item ${
              page === "system" ? "nav-item-active" : ""
            }`}
            onClick={() => navigate("system")}
          >
            <span className="nav-icon">⚙</span>
            <span>System Status</span>
          </button>
        </nav>

        <div className="sidebar-footer">
          <div className="security-badge">
            <div className="security-icon">✓</div>

            <div>
              <strong>SECURE SESSION</strong>
              <span>Local forensic workspace</span>
            </div>
          </div>

          <div className="version-label">VERITAS ENGINE • v1.0</div>

          <button className="logout-button" onClick={onLogout}>
            <span>⇥</span>
            <span>Logout</span>
          </button>
        </div>
      </aside>

      {sidebarOpen && (
        <div
          className="sidebar-overlay"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <main className="main-content">
        <header className="top-header">
          <button
            className="mobile-menu-button"
            onClick={() => setSidebarOpen((value) => !value)}
          >
            ☰
          </button>

          <div className="header-breadcrumb">
            <span>VERITAS</span>
            <span className="breadcrumb-separator">/</span>
            <strong>
              {page === "dashboard" && "Dashboard"}
              {page === "investigation" && "New Investigation"}
              {page === "history" && "Investigation History"}
              {page === "reports" && "Reports"}
              {page === "modules" && "Forensic Modules"}
              {page === "system" && "System Status"}
            </strong>
          </div>

          <div className="header-actions">
            <div className="engine-status">
              <span
                className={`status-dot ${
                  backendOnline ? "status-online" : "status-offline"
                }`}
              />
              <span>
                {backendOnline ? "ENGINE ONLINE" : "ENGINE OFFLINE"}
              </span>
            </div>

            <div className="user-badge">
              <div className="user-avatar">VR</div>

              <div className="user-info">
                <strong>Forensic Analyst</strong>
                <span>Authenticated Session</span>
              </div>
            </div>
          </div>
        </header>

        <div className="content-container">
          {page === "dashboard" && (
            <DashboardPage
              backendOnline={backendOnline}
              history={history}
              activeModules={activeModuleCount}
              plannedModules={plannedModuleCount}
              navigate={navigate}
              checkBackend={checkBackend}
            />
          )}

          {page === "investigation" && (
            <InvestigationPage
              selectedFile={selectedFile}
              previewUrl={previewUrl}
              result={result}
              isAnalyzing={isAnalyzing}
              isDragging={isDragging}
              backendOnline={backendOnline}
              onFileChange={handleFileChange}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              clearInvestigation={clearInvestigation}
              formatPercent={formatPercent}
              getConfidencePercent={getConfidencePercent}
              verdictClass={renderVerdictClass()}
              getVerdict={getVerdict}
            />
          )}

          {page === "history" && (
            <HistoryPage history={history} formatPercent={formatPercent} />
          )}

          {page === "reports" && (
            <ReportsPage history={history} result={result} />
          )}

          {page === "modules" && (
            <ModulesPage
              activeModules={activeModuleCount}
              plannedModules={plannedModuleCount}
            />
          )}

          {page === "system" && (
            <SystemPage
              backendOnline={backendOnline}
              checkBackend={checkBackend}
            />
          )}
        </div>
      </main>
    </div>
  );
}

function PageHeader({
  eyebrow,
  title,
  description,
  icon,
}: {
  eyebrow: string;
  title: string;
  description: string;
  icon: string;
}) {
  return (
    <section className="page-heading-section">
      <div className="page-heading-icon">{icon}</div>

      <div>
        <div className="page-eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
    </section>
  );
}

function DashboardPage({
  backendOnline,
  history,
  activeModules,
  plannedModules,
  navigate,
  checkBackend,
}: {
  backendOnline: boolean;
  history: HistoryItem[];
  activeModules: number;
  plannedModules: number;
  navigate: (page: Page) => void;
  checkBackend: () => Promise<void>;
}) {
  const recentAnalyses = history.slice(0, 4);

  return (
    <div className="page-section">
      <section className="hero-section">
        <div className="hero-copy">
          <div className="hero-kicker">DIGITAL MEDIA FORENSICS PLATFORM</div>

          <h1>
            Intelligence for
            <br />
            <span>Digital Evidence.</span>
          </h1>

          <p>
            VERITAS provides an evidence-oriented workspace for analyzing
            digital media, detecting facial manipulation, and documenting
            forensic findings.
          </p>

          <div className="hero-actions">
            <button
              className="primary-button"
              onClick={() => navigate("investigation")}
            >
              <span>⊕</span>
              Start Investigation
            </button>

            <button
              className="secondary-button"
              onClick={() => navigate("modules")}
            >
              View Forensic Modules
              <span>→</span>
            </button>
          </div>
        </div>

        <div className="hero-visual">
          <div className="forensic-ring ring-one" />
          <div className="forensic-ring ring-two" />
          <div className="forensic-ring ring-three" />

          <div className="hero-core">
            <div className="hero-core-letter">V</div>
            <span>
              FORENSIC
              <br />
              ENGINE
            </span>
          </div>

          <div className="scan-line" />
        </div>
      </section>

      <section className="metrics-grid">
        <MetricCard
          label="ENGINE STATUS"
          value={backendOnline ? "ONLINE" : "OFFLINE"}
          detail={
            backendOnline
              ? "FastAPI inference service responding"
              : "Start the ML service on port 8000"
          }
          icon="◉"
          status={backendOnline ? "good" : "warning"}
        />

        <MetricCard
          label="ANALYSES"
          value={String(history.length)}
          detail={
            history.length === 0
              ? "No investigations in this session"
              : "Current workspace investigations"
          }
          icon="⌁"
          status="neutral"
        />

        <MetricCard
          label="ACTIVE MODULES"
          value={String(activeModules)}
          detail={`${plannedModules} additional modules planned`}
          icon="◆"
          status="good"
        />

        <MetricCard
          label="MODEL"
          value="EfficientNet"
          detail="B0 classification backbone"
          icon="◇"
          status="neutral"
        />
      </section>

      <section className="dashboard-grid">
        <div className="dashboard-card pipeline-card">
          <div className="card-heading">
            <div>
              <div className="card-eyebrow">ANALYSIS PIPELINE</div>
              <h2>Forensic Processing Stages</h2>
            </div>

            <span className="live-label">
              ● {backendOnline ? "LIVE" : "OFFLINE"}
            </span>
          </div>

          <div className="pipeline-list">
            <PipelineItem
              number="01"
              title="Media Intake"
              description="Validate and decode submitted media."
              status="ready"
            />

            <PipelineItem
              number="02"
              title="Face Detection"
              description="Locate facial regions using the detection engine."
              status="ready"
            />

            <PipelineItem
              number="03"
              title="Deepfake Classification"
              description="Evaluate facial imagery with the trained model."
              status="ready"
            />

            <PipelineItem
              number="04"
              title="Evidence Reporting"
              description="Present confidence and forensic observations."
              status="planned"
            />
          </div>
        </div>

        <div className="dashboard-card quick-actions-card">
          <div className="card-eyebrow">QUICK ACCESS</div>
          <h2>Investigation Tools</h2>

          <div className="quick-action-list">
            <QuickAction
              icon="⊕"
              title="Analyze Media"
              description="Upload an image for forensic analysis."
              onClick={() => navigate("investigation")}
            />

            <QuickAction
              icon="◷"
              title="View History"
              description="Review analyses from this session."
              onClick={() => navigate("history")}
            />

            <QuickAction
              icon="◈"
              title="Forensic Modules"
              description="Review available analysis capabilities."
              onClick={() => navigate("modules")}
            />

            <QuickAction
              icon="⚙"
              title="System Status"
              description="Check service and model readiness."
              onClick={() => navigate("system")}
            />
          </div>
        </div>
      </section>

      <section className="dashboard-card recent-analysis-card">
        <div className="card-heading">
          <div>
            <div className="card-eyebrow">RECENT ANALYSIS</div>
            <h2>Investigation Activity</h2>
          </div>

          <button
            className="text-button"
            onClick={() => navigate("history")}
          >
            View History →
          </button>
        </div>

        {recentAnalyses.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-icon">◷</div>

            <h2>No investigations yet</h2>

            <p>
              Completed media analyses will appear here after you run an
              investigation.
            </p>

            <button
              className="secondary-button"
              onClick={() => navigate("investigation")}
            >
              Start First Analysis
              <span>→</span>
            </button>
          </div>
        ) : (
          <div className="recent-analysis-list">
            {recentAnalyses.map((item) => (
              <div className="recent-analysis-item" key={item.id}>
                <div className="recent-analysis-file">
                  <div className="recent-analysis-icon">◇</div>

                  <div>
                    <strong>{item.filename}</strong>
                    <span>{item.timestamp}</span>
                  </div>
                </div>

                <div className="recent-analysis-verdict">
                  <span className="card-eyebrow">VERDICT</span>

                  <strong
                    className={
                      item.verdict === "FAKE"
                        ? "recent-verdict-fake"
                        : item.verdict === "REAL"
                          ? "recent-verdict-real"
                          : ""
                    }
                  >
                    {item.verdict}
                  </strong>
                </div>

                <div className="recent-analysis-confidence">
                  <span className="card-eyebrow">CONFIDENCE</span>
                  <strong>{formatPercentValue(item.confidence)}</strong>
                </div>

                <div className="recent-analysis-faces">
                  <span className="card-eyebrow">FACES</span>
                  <strong>{item.faces}</strong>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="dashboard-card">
        <div className="card-heading">
          <div>
            <div className="card-eyebrow">FORENSIC CAPABILITIES</div>
            <h2>Analysis Modules</h2>
          </div>

          <button
            className="text-button"
            onClick={() => navigate("modules")}
          >
            View All Modules →
          </button>
        </div>

        <div className="dashboard-module-overview">
          <div className="dashboard-module-status">
            <div className="dashboard-module-number">{activeModules}</div>

            <div>
              <strong>Active Modules</strong>
              <span>Currently available for analysis</span>
            </div>
          </div>

          <div className="dashboard-module-status">
            <div className="dashboard-module-number">{plannedModules}</div>

            <div>
              <strong>Planned Modules</strong>
              <span>Additional forensic capabilities</span>
            </div>
          </div>

          <div className="dashboard-module-status">
            <div className="dashboard-module-number">
              {activeModules + plannedModules}
            </div>

            <div>
              <strong>Total Capabilities</strong>
              <span>Current and planned forensic modules</span>
            </div>
          </div>
        </div>
      </section>

      <section className="dashboard-card integrity-card">
        <div className="integrity-icon">✓</div>

        <div className="integrity-copy">
          <div className="card-eyebrow">EVIDENCE HANDLING</div>

          <h2>Analysis-first forensic workflow</h2>

          <p>
            VERITAS records analysis metadata and separates implemented
            forensic signals from planned capabilities. No unavailable
            forensic measurement is presented as an actual result.
          </p>
        </div>

        <button
          className="text-button"
          onClick={() => void checkBackend()}
        >
          Recheck Engine →
        </button>
      </section>
    </div>
  );
}

function InvestigationPage({
  selectedFile,
  previewUrl,
  result,
  isAnalyzing,
  isDragging,
  backendOnline,
  onFileChange,
  onDrop,
  onDragOver,
  onDragLeave,
  clearInvestigation,
  formatPercent,
  getConfidencePercent,
  verdictClass,
  getVerdict,
}: {
  selectedFile: File | null;
  previewUrl: string;
  result: AnalysisResult | null;
  isAnalyzing: boolean;
  isDragging: boolean;
  backendOnline: boolean;
  onFileChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onDrop: (event: DragEvent<HTMLDivElement>) => void;
  onDragOver: (event: DragEvent<HTMLDivElement>) => void;
  onDragLeave: (event: DragEvent<HTMLDivElement>) => void;
  clearInvestigation: () => void;
  formatPercent: (value?: number | null) => string;
  getConfidencePercent: (value?: number | null) => number;
  verdictClass: string;
  getVerdict: () => string;
}) {
  const prediction =
    result?.model?.prediction ??
    result?.prediction ??
    result?.verdict ??
    getVerdict();

  const facesDetected =
    result?.face_detection?.faces_detected ??
    result?.face_detection?.count ??
    result?.faces_detected ??
    0;

  const modelConfidence =
    result?.model?.confidence ?? result?.confidence;

  const realProbability =
    result?.model?.real_probability ?? result?.real_probability;

  const fakeProbability =
    result?.model?.fake_probability ?? result?.fake_probability;

  const imageWidth = result?.image?.width ?? result?.image_width;
  const imageHeight = result?.image?.height ?? result?.image_height;

  const modelName =
    result?.model?.name ?? result?.model_name ?? "EfficientNet-B0";

  const modelVersion = result?.model_version ?? "Current build";

  const modelReady =
    result?.model?.trained_for_deepfake_detection ??
    result?.trained_for_deepfake_detection ??
    false;

  const explainabilityStatus =
    result?.explainability?.status ?? result?.explainability_status;

  const analysisComplete = Boolean(
    result && !result.error && !isAnalyzing
  );

  const metadata = result?.metadata;

  const metadataFields = metadata?.fields
    ? Object.entries(metadata.fields)
    : [];

  return (
    <div className="page-section">
      <PageHeader
        eyebrow="FORENSIC INVESTIGATION"
        title="Media Analysis Workspace"
        description="Submit digital evidence, execute the forensic pipeline, and inspect the available analytical findings."
        icon="⊕"
      />

      <div className="workspace-toolbar">
        <div className="workspace-state">
          <span
            className={`status-dot ${
              backendOnline ? "status-online" : "status-offline"
            }`}
          />

          <span>
            {backendOnline
              ? "Analysis engine available"
              : "Analysis engine unavailable"}
          </span>

          <span className="workspace-divider">•</span>

          <span>
            {selectedFile ? "Evidence loaded" : "Awaiting evidence"}
          </span>
        </div>

        {selectedFile && (
          <button
            className="secondary-button small-button"
            onClick={clearInvestigation}
          >
            Clear Investigation
          </button>
        )}
      </div>

      <section className="dashboard-card investigation-pipeline-card">
        <div className="panel-header">
          <div>
            <div className="card-eyebrow">FORENSIC PROCESSING PIPELINE</div>
            <h2>Evidence Analysis Stages</h2>
          </div>

          <div
            className={`pipeline-status ${
              isAnalyzing
                ? "pipeline-processing"
                : analysisComplete
                  ? "pipeline-complete"
                  : "pipeline-ready"
            }`}
          >
            <span className="status-dot status-online" />

            {isAnalyzing
              ? "PROCESSING"
              : analysisComplete
                ? "COMPLETE"
                : "READY"}
          </div>
        </div>

        <div className="pipeline-large">
          <div className="pipeline-step">
            <div className="pipeline-number">01</div>
            <div className="pipeline-content">
              <strong>Evidence Intake</strong>
              <span>File validation and media decoding</span>
            </div>
          </div>

          <div className="pipeline-step">
            <div className="pipeline-number">02</div>
            <div className="pipeline-content">
              <strong>Metadata Analysis</strong>
              <span>Inspect embedded image properties and EXIF data</span>
            </div>
          </div>

          <div className="pipeline-step">
            <div className="pipeline-number">03</div>
            <div className="pipeline-content">
              <strong>Face Detection</strong>
              <span>Locate facial regions for analysis</span>
            </div>
          </div>

          <div className="pipeline-step">
            <div className="pipeline-number">04</div>
            <div className="pipeline-content">
              <strong>AI Classification</strong>
              <span>Evaluate facial imagery with the trained model</span>
            </div>
          </div>

          <div className="pipeline-step">
            <div className="pipeline-number">05</div>
            <div className="pipeline-content">
              <strong>Evidence Output</strong>
              <span>Consolidate available analytical findings</span>
            </div>
          </div>
        </div>
      </section>

      <div className="investigation-layout">
        <section className="intake-panel">
          <div className="panel-header">
            <div>
              <div className="card-eyebrow">EVIDENCE INTAKE</div>
              <h2>Submit Media</h2>
            </div>

            <div className="secure-label">SECURE</div>
          </div>

          <div
            className={`large-drop-zone ${
              isDragging ? "drop-zone-active" : ""
            } ${selectedFile ? "drop-zone-has-file" : ""}`}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
          >
            {!selectedFile ? (
              <>
                <div className="drop-zone-symbol">↑</div>

                <h3>Drop evidence here</h3>

                <p>
                  Drag and drop an image into this area, or select a file
                  manually.
                </p>

                <label className="upload-button">
                  Select Evidence

                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    onChange={onFileChange}
                    hidden
                  />
                </label>

                <div className="file-format-note">JPEG • PNG • WEBP</div>
              </>
            ) : (
              <>
                <div className="file-selected-icon">✓</div>

                <h3>Evidence loaded</h3>

                <p className="selected-file-name">{selectedFile.name}</p>

                <p>
                  {formatSafeNumber(
                    selectedFile.size / 1024 / 1024,
                    2
                  )}{" "}
                  MB
                </p>

                <label className="secondary-button file-change-button">
                  Change File

                  <input
                    type="file"
                    accept="image/jpeg,image/png,image/webp"
                    onChange={onFileChange}
                    hidden
                  />
                </label>
              </>
            )}
          </div>

          <div className="intake-notice">
            <span>ⓘ</span>

            <p>
              Current engine supports image-based facial analysis. JPEG, PNG,
              and WebP files are accepted.
            </p>
          </div>
        </section>

        <section className="analysis-preview-panel">
          <div className="panel-header">
            <div>
              <div className="card-eyebrow">EVIDENCE PREVIEW</div>
              <h2>Submitted Asset</h2>
            </div>

            {selectedFile && (
              <span className="asset-status">LOADED</span>
            )}
          </div>

          <div className="analysis-preview">
            {previewUrl ? (
              <img
                src={previewUrl}
                alt="Submitted forensic evidence"
              />
            ) : (
              <div className="empty-preview">
                <div className="empty-preview-icon">◇</div>
                <span>Evidence preview will appear here</span>
              </div>
            )}
          </div>

          {selectedFile && (
            <div className="asset-metadata">
              <MetadataRow label="Filename" value={selectedFile.name} />

              <MetadataRow label="MIME Type" value={selectedFile.type} />

              <MetadataRow
                label="File Size"
                value={`${formatSafeNumber(
                  selectedFile.size / 1024 / 1024,
                  2
                )} MB`}
              />

              <MetadataRow
                label="Analysis State"
                value={
                  isAnalyzing
                    ? "PROCESSING"
                    : result
                      ? "COMPLETE"
                      : "READY"
                }
              />
            </div>
          )}
        </section>
      </div>

      {isAnalyzing && (
        <section className="processing-panel">
          <div className="processing-spinner" />

          <div>
            <div className="card-eyebrow">FORENSIC ENGINE</div>

            <h2>Analyzing evidence…</h2>

            <p>
              Decoding image, extracting metadata, detecting facial regions,
              and executing classification.
            </p>
          </div>

          <div className="processing-stage-list">
            <span>01 Intake</span>
            <span>02 Metadata</span>
            <span>03 Face Detection</span>
            <span>04 Classification</span>
            <span>05 Evidence Output</span>
          </div>
        </section>
      )}

      {result && !isAnalyzing && (
        <section className="results-area">
          <div className="results-header">
            <div>
              <div className="card-eyebrow">FORENSIC RESULT</div>
              <h2>Analysis Findings</h2>
            </div>

            <div className="result-timestamp">Analysis complete</div>
          </div>

          <section className="dashboard-card case-record-card">
            <div className="panel-header">
              <div>
                <div className="card-eyebrow">EVIDENCE SUMMARY</div>

                <h2>VERITAS Case Record</h2>

                <p className="panel-description">
                  Persistent analysis identity and key evidence attributes
                  associated with the completed forensic examination.
                </p>
              </div>

              <span className="module-status module-status-active">
                ANALYZED
              </span>
            </div>

            <div className="metadata-forensics-grid">
              <div className="metadata-forensics-item">
                <span>CASE ID</span>
                <strong>
                  {result.analysis_id != null
                    ? `VERITAS-${result.analysis_id}`
                    : "Pending"}
                </strong>
              </div>

              <div className="metadata-forensics-item">
                <span>EVIDENCE FILE</span>
                <strong>
                  {result.filename || selectedFile?.name || "—"}
                </strong>
              </div>

              <div className="metadata-forensics-item">
                <span>VERDICT</span>
                <strong>
                  {prediction
                    ? String(prediction).toUpperCase()
                    : "UNDETERMINED"}
                </strong>
              </div>

              <div className="metadata-forensics-item">
                <span>MODEL CONFIDENCE</span>
                <strong>{formatPercent(modelConfidence)}</strong>
              </div>

              <div className="metadata-forensics-item">
                <span>FACES ANALYZED</span>
                <strong>{String(facesDetected)}</strong>
              </div>

              <div className="metadata-forensics-item">
                <span>MODEL</span>
                <strong>{modelName}</strong>
              </div>

              <div className="metadata-forensics-item">
                <span>MODEL VERSION</span>
                <strong>{modelVersion}</strong>
              </div>

              <div className="metadata-forensics-item">
                <span>PROCESSING TIME</span>
                <strong>
                  {result.processing_time_ms != null
                    ? `${formatSafeNumber(
                        result.processing_time_ms,
                        0
                      )} ms`
                    : "—"}
                </strong>
              </div>

              <div className="metadata-forensics-item">
                <span>IMAGE DIMENSIONS</span>
                <strong>
                  {imageWidth != null && imageHeight != null
                    ? `${imageWidth} × ${imageHeight}`
                    : "—"}
                </strong>
              </div>

              <div className="metadata-forensics-item">
                <span>SHA-256 EVIDENCE HASH</span>
                <strong
                  title={result.image_integrity?.sha256 || undefined}
                >
                  {result.image_integrity?.sha256 || "Unavailable"}
                </strong>
              </div>

              <div className="metadata-forensics-item">
                <span>INTEGRITY STATUS</span>
                <strong>
                  {result.image_integrity?.status || "Not reported"}
                </strong>
              </div>

              <div className="metadata-forensics-item">
                <span>SESSION TIMESTAMP</span>
                <strong>{new Date().toLocaleString()}</strong>
              </div>
            </div>

            <div className="intake-notice">
              <span>ⓘ</span>

              <p>
                The case ID is linked to the persisted analysis record. The
                SHA-256 value is the evidence hash returned by the integrity
                module. Forensic indicators are supporting evidence and should
                be interpreted together with the primary classification.
              </p>
            </div>
          </section>

          {result.error ? (
            <div className="error-result-panel">
              <div className="error-icon">!</div>

              <div>
                <h3>Analysis failed</h3>
                <p>{result.error}</p>
              </div>
            </div>
          ) : (
            <>
              <div className="verdict-layout">
                <div className={`verdict-panel ${verdictClass}`}>
                  <div className="verdict-label">CLASSIFICATION</div>

                  <div className="verdict-value">{getVerdict()}</div>

                  <div className="verdict-description">
                    {getVerdict() === "FAKE"
                      ? "The current model classified the analyzed facial region as fake."
                      : getVerdict() === "REAL"
                        ? "The current model classified the analyzed facial region as real."
                        : "The model returned an undetermined classification."}
                  </div>
                </div>

                <div className="confidence-panel">
                  <div className="result-card-title">MODEL CONFIDENCE</div>

                  <div className="confidence-number">
                    {formatPercent(modelConfidence)}
                  </div>

                  <div className="confidence-bar">
                    <div
                      className="confidence-fill"
                      style={{
                        width: `${Math.min(
                          100,
                          Math.max(
                            0,
                            getConfidencePercent(modelConfidence)
                          )
                        )}%`,
                      }}
                    />
                  </div>

                  <p>
                    Confidence represents the classification model's output
                    for this analysis.
                  </p>
                </div>
              </div>

              <div className="result-metrics-grid">
                <ResultMetric
                  label="REAL PROBABILITY"
                  value={formatPercent(realProbability)}
                />

                <ResultMetric
                  label="FAKE PROBABILITY"
                  value={formatPercent(fakeProbability)}
                />

                <ResultMetric
                  label="FACES DETECTED"
                  value={String(facesDetected)}
                />

                <ResultMetric
                  label="PROCESSING TIME"
                  value={
                    result.processing_time_ms !== undefined
                      ? `${formatSafeNumber(
                          result.processing_time_ms,
                          0
                        )} ms`
                      : "—"
                  }
                />

                <ResultMetric
                  label="IMAGE DIMENSIONS"
                  value={
                    imageWidth && imageHeight
                      ? `${imageWidth} × ${imageHeight}`
                      : "—"
                  }
                />

                <ResultMetric
                  label="MODEL"
                  value={modelName || "EfficientNet-B0"}
                />
              </div>

              <div className="forensic-disclosure-grid">
                <div className="disclosure-card">
                  <div className="disclosure-icon">◉</div>

                  <div>
                    <div className="card-eyebrow">FACE DETECTION</div>

                    <h3>
                      {facesDetected} facial region
                      {facesDetected === 1 ? "" : "s"}
                    </h3>

                    <p>
                      Detection method:{" "}
                      {result.face_detection?.method ||
                        "OpenCV Haar Cascade"}
                    </p>
                  </div>
                </div>

                <div className="disclosure-card">
                  <div className="disclosure-icon">✦</div>

                  <div>
                    <div className="card-eyebrow">EXPLAINABILITY</div>

                    <h3>{explainabilityStatus || "Not generated"}</h3>

                    <p>
                      Method:{" "}
                      {result.explainability?.method ||
                        result.explainability_method ||
                        "Grad-CAM"}
                    </p>
                  </div>
                </div>

                <div className="disclosure-card">
                  <div className="disclosure-icon">◆</div>

                  <div>
                    <div className="card-eyebrow">MODEL STATE</div>

                    <h3>
                      {modelReady
                        ? "Trained model"
                        : "Training status not reported"}
                    </h3>

                    <p>
                      Model version:{" "}
                      {result.model_version || "Current build"}
                    </p>
                  </div>
                </div>
              </div>

              <section className="dashboard-card gradcam-card">
                <div className="panel-header">
                  <div>
                    <div className="card-eyebrow">FORENSIC MODULE 04</div>

                    <h2>Grad-CAM Explainability</h2>

                    <p className="panel-description">
                      Visual explanation highlighting the facial regions that
                      influenced the EfficientNet-B0 classification.
                    </p>
                  </div>

                  <span
                    className={`module-status ${
                      explainabilityStatus === "generated"
                        ? "module-status-active"
                        : explainabilityStatus === "error"
                          ? "module-status-warning"
                          : "module-status-neutral"
                    }`}
                  >
                    {explainabilityStatus === "generated"
                      ? "GENERATED"
                      : explainabilityStatus === "error"
                        ? "ERROR"
                        : "PENDING"}
                  </span>
                </div>

                {result.explainability?.heatmap_base64 ? (
                  <div className="gradcam-content">
                    <div className="gradcam-visual">
                      <img
                        src={`data:image/png;base64,${result.explainability.heatmap_base64}`}
                        alt="Grad-CAM explanation heatmap"
                        className="gradcam-image"
                      />
                    </div>

                    <div className="gradcam-details">
                      <div className="gradcam-detail-block">
                        <span className="card-eyebrow">
                          EXPLANATION METHOD
                        </span>

                        <strong>
                          {result.explainability.method || "Grad-CAM"}
                        </strong>

                        <p>
                          The highlighted regions indicate areas that
                          contributed most strongly to the model's selected
                          classification.
                        </p>
                      </div>

                      <div className="gradcam-detail-block">
                        <span className="card-eyebrow">TARGET CLASS</span>

                        <strong>
                          {(prediction || "UNKNOWN").toUpperCase()}
                        </strong>

                        <p>
                          Heatmap generated for the predicted class using the
                          selected face region.
                        </p>
                      </div>

                      <div className="gradcam-note">
                        <span>ⓘ</span>

                        <p>
                          Grad-CAM is an interpretability aid. It shows where
                          the model focused; it is not independent proof that
                          an image is authentic or manipulated.
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="gradcam-empty">
                    <div className="gradcam-empty-icon">✦</div>

                    <div>
                      <h3>
                        {explainabilityStatus === "error"
                          ? "Grad-CAM generation failed"
                          : "Grad-CAM explanation not available"}
                      </h3>

                      <p>
                        {explainabilityStatus === "error"
                          ? "The forensic classification completed, but the visual explanation could not be generated for this analysis."
                          : "Run a new analysis to generate the visual explanation for the selected face."}
                      </p>
                    </div>
                  </div>
                )}
              </section>

              <section className="dashboard-card metadata-forensics-card">
                <div className="panel-header">
                  <div>
                    <div className="card-eyebrow">FORENSIC MODULE 05</div>

                    <h2>Metadata / EXIF Analysis</h2>

                    <p className="panel-description">
                      Inspect embedded image properties and metadata returned
                      by the forensic engine.
                    </p>
                  </div>

                  <span
                    className={`module-status ${
                      metadata?.exif_available
                        ? "module-status-active"
                        : "module-status-planned"
                    }`}
                  >
                    {metadata?.exif_available
                      ? "EXIF DETECTED"
                      : "NO EXIF"}
                  </span>
                </div>

                {!metadata ? (
                  <div className="metadata-unavailable">
                    <div className="metadata-unavailable-icon">!</div>

                    <div>
                      <h3>Metadata analysis unavailable</h3>

                      <p>
                        The forensic engine did not return metadata information
                        for this analysis.
                      </p>
                    </div>
                  </div>
                ) : (
                  <>
                    <div className="metadata-forensics-grid">
                      <div className="metadata-forensics-item">
                        <span>IMAGE FORMAT</span>
                        <strong>{metadata.format || "—"}</strong>
                      </div>

                      <div className="metadata-forensics-item">
                        <span>EXIF STATUS</span>
                        <strong>
                          {metadata.exif_available
                            ? "AVAILABLE"
                            : "NOT PRESENT"}
                        </strong>
                      </div>

                      <div className="metadata-forensics-item">
                        <span>CAMERA MAKE</span>
                        <strong>
                          {metadata.camera_make || "Not available"}
                        </strong>
                      </div>

                      <div className="metadata-forensics-item">
                        <span>CAMERA MODEL</span>
                        <strong>
                          {metadata.camera_model || "Not available"}
                        </strong>
                      </div>

                      <div className="metadata-forensics-item">
                        <span>DATE / TIME</span>
                        <strong>
                          {metadata.date_time || "Not available"}
                        </strong>
                      </div>

                      <div className="metadata-forensics-item">
                        <span>SOFTWARE</span>
                        <strong>
                          {metadata.software || "Not available"}
                        </strong>
                      </div>

                      <div className="metadata-forensics-item">
                        <span>ORIENTATION</span>
                        <strong>
                          {metadata.orientation !== undefined &&
                          metadata.orientation !== null
                            ? String(metadata.orientation)
                            : "Not available"}
                        </strong>
                      </div>

                      <div className="metadata-forensics-item">
                        <span>GPS DATA</span>
                        <strong>
                          {metadata.gps_present
                            ? "PRESENT"
                            : "NOT PRESENT"}
                        </strong>
                      </div>
                    </div>

                    {metadataFields.length > 0 && (
                      <div className="metadata-fields-section">
                        <div className="card-eyebrow">
                          EMBEDDED METADATA FIELDS
                        </div>

                        <div className="metadata-fields-list">
                          {metadataFields.map(([key, value]) => (
                            <MetadataRow
                              key={key}
                              label={key}
                              value={
                                value === null || value === undefined
                                  ? "—"
                                  : String(value)
                              }
                            />
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="metadata-forensics-note">
                      <span>ⓘ</span>

                      <p>
                        Metadata presence or absence is an investigative
                        observation, not by itself proof that an image is
                        authentic or manipulated. Metadata can be removed,
                        modified, or regenerated by image processing software.
                      </p>
                    </div>
                  </>
                )}
              </section>

              <section className="dashboard-card forensic-analysis-results-card noise-frequency-dashboard noise-frequency-reference">
                <div className="panel-header">
                  <div>
                    <div className="card-eyebrow">FORENSIC MODULE 08</div>

                    <h2>Noise &amp; Frequency Analysis</h2>

                    <p className="panel-description">
                      Analyzes image noise patterns and frequency components to
                      detect manipulation artifacts.
                    </p>
                  </div>

                  <span className="module-status module-active">
                    {result.noise_frequency_analysis?.status === "PASS"
                      ? "ACTIVE"
                      : result.noise_frequency_analysis?.status || "N/A"}
                  </span>
                </div>

                <div className="noise-frequency-domain-panel">
                  <div className="noise-frequency-section-heading">
                    <h3>Frequency Domain Analysis</h3>
                  </div>

                  <div className="noise-frequency-reference-visual-grid">
                    <div className="noise-frequency-reference-card">
                      <div className="noise-frequency-reference-title">
                        Low-Frequency Component
                      </div>

                      <div className="noise-frequency-reference-image-frame">
                        {result.noise_frequency_analysis
                          ?.low_frequency_base64 ? (
                          <img
                            src={`data:image/png;base64,${result.noise_frequency_analysis.low_frequency_base64}`}
                            alt="Low-frequency component"
                            className="noise-frequency-reference-image"
                          />
                        ) : (
                          <div className="noise-frequency-reference-empty">
                            Low-frequency component unavailable
                          </div>
                        )}
                      </div>

                      <p className="noise-frequency-reference-description">
                        Represents smooth, large-scale image structures.
                      </p>
                    </div>

                    <div className="noise-frequency-reference-card">
                      <div className="noise-frequency-reference-title">
                        High-Frequency Component
                      </div>

                      <div className="noise-frequency-reference-image-frame">
                        {result.noise_frequency_analysis
                          ?.high_frequency_base64 ? (
                          <img
                            src={`data:image/png;base64,${result.noise_frequency_analysis.high_frequency_base64}`}
                            alt="High-frequency component"
                            className="noise-frequency-reference-image"
                          />
                        ) : (
                          <div className="noise-frequency-reference-empty">
                            High-frequency component unavailable
                          </div>
                        )}
                      </div>

                      <p className="noise-frequency-reference-description">
                        Highlights edges, fine details and high-frequency
                        variations.
                      </p>
                    </div>

                    <div className="noise-frequency-reference-card">
                      <div className="noise-frequency-reference-title">
                        Noise Map
                      </div>

                      <div className="noise-frequency-reference-image-frame">
                        {result.noise_frequency_analysis
                          ?.noise_map_base64 ? (
                          <img
                            src={`data:image/png;base64,${result.noise_frequency_analysis.noise_map_base64}`}
                            alt="Noise map"
                            className="noise-frequency-reference-image"
                          />
                        ) : (
                          <div className="noise-frequency-reference-empty">
                            Noise map unavailable
                          </div>
                        )}
                      </div>

                      <p className="noise-frequency-reference-description">
                        Visualizes local noise variation across the image.
                      </p>
                    </div>

                    <div className="noise-frequency-reference-card">
                      <div className="noise-frequency-reference-title">
                        FFT Frequency Spectrum
                      </div>

                      <div className="noise-frequency-reference-image-frame">
                        {result.noise_frequency_analysis
                          ?.fft_spectrum_base64 ? (
                          <img
                            src={`data:image/png;base64,${result.noise_frequency_analysis.fft_spectrum_base64}`}
                            alt="FFT frequency spectrum"
                            className="noise-frequency-reference-image"
                          />
                        ) : (
                          <div className="noise-frequency-reference-empty">
                            FFT spectrum unavailable
                          </div>
                        )}
                      </div>

                      <p className="noise-frequency-reference-description">
                        Shows the distribution of frequency energy in the
                        image.
                      </p>
                    </div>

                    <div className="noise-frequency-reference-card">
                      <div className="noise-frequency-reference-title">
                        Noise Level Heatmap
                      </div>

                      <div className="noise-frequency-reference-image-frame">
                        {result.noise_frequency_analysis
                          ?.noise_heatmap_base64 ? (
                          <img
                            src={`data:image/png;base64,${result.noise_frequency_analysis.noise_heatmap_base64}`}
                            alt="Noise level heatmap"
                            className="noise-frequency-reference-image"
                          />
                        ) : (
                          <div className="noise-frequency-reference-empty">
                            Noise heatmap unavailable
                          </div>
                        )}
                      </div>

                      <p className="noise-frequency-reference-description">
                        Highlights regions with stronger noise-level
                        variations.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="noise-frequency-reference-bottom-grid">
                  <div className="noise-frequency-metrics-panel">
                    <div className="noise-frequency-section-heading">
                      <h3>Noise Analysis Metrics</h3>
                    </div>

                    <div className="noise-frequency-reference-metrics-grid">
                      <div className="noise-frequency-reference-metric">
                        <span className="noise-frequency-reference-metric-icon">
                          ∿
                        </span>
                        <span>Noise Variance</span>
                        <strong>
                          {result.noise_frequency_analysis
                            ?.noise_variance != null
                            ? formatSafeNumber(
                                result.noise_frequency_analysis
                                  .noise_variance,
                                2
                              )
                            : "—"}
                        </strong>
                      </div>

                      <div className="noise-frequency-reference-metric">
                        <span className="noise-frequency-reference-metric-icon">
                          〽
                        </span>
                        <span>Noise Std. Deviation</span>
                        <strong>
                          {result.noise_frequency_analysis
                            ?.noise_standard_deviation != null
                            ? formatSafeNumber(
                                result.noise_frequency_analysis
                                  .noise_standard_deviation,
                                2
                              )
                            : "—"}
                        </strong>
                      </div>

                      <div className="noise-frequency-reference-metric">
                        <span className="noise-frequency-reference-metric-icon">
                          △
                        </span>
                        <span>Sharpness Variance</span>
                        <strong>
                          {result.noise_frequency_analysis
                            ?.sharpness_variance != null
                            ? formatSafeNumber(
                                result.noise_frequency_analysis
                                  .sharpness_variance,
                                2
                              )
                            : "—"}
                        </strong>
                      </div>

                      <div className="noise-frequency-reference-metric">
                        <span className="noise-frequency-reference-metric-icon">
                          ▥
                        </span>
                        <span>Frequency Energy</span>
                        <strong>
                          {result.noise_frequency_analysis
                            ?.frequency_energy != null
                            ? formatSafeNumber(
                                result.noise_frequency_analysis
                                  .frequency_energy,
                                2
                              )
                            : "—"}
                        </strong>
                      </div>

                      <div className="noise-frequency-reference-metric">
                        <span className="noise-frequency-reference-metric-icon">
                          ◌
                        </span>
                        <span>High-Frequency Ratio</span>
                        <strong>
                          {result.noise_frequency_analysis
                            ?.high_frequency_energy_ratio != null
                            ? formatPercentValue(
                                result.noise_frequency_analysis
                                  .high_frequency_energy_ratio
                              )
                            : "—"}
                        </strong>
                      </div>
                    </div>
                  </div>

                  <div className="noise-frequency-reference-summary">
                    <div className="noise-frequency-section-heading noise-frequency-summary-heading">
                      <h3>Analysis Summary</h3>

                      <span className="noise-frequency-summary-status">
                        {result.noise_frequency_analysis?.status || "N/A"}
                      </span>
                    </div>

                    <div className="noise-frequency-summary-row">
                      <span>Noise Level</span>
                      <strong>
                        {result.noise_frequency_analysis?.noise_level || "—"}
                      </strong>
                    </div>

                    <div className="noise-frequency-summary-row">
                      <span>Frequency Energy</span>
                      <strong>
                        {result.noise_frequency_analysis
                          ?.frequency_energy != null
                          ? formatSafeNumber(
                              result.noise_frequency_analysis
                                .frequency_energy,
                              2
                            )
                          : "—"}
                      </strong>
                    </div>

                    <div className="noise-frequency-summary-row">
                      <span>High-Frequency Ratio</span>
                      <strong>
                        {result.noise_frequency_analysis
                          ?.high_frequency_energy_ratio != null
                          ? formatPercentValue(
                              result.noise_frequency_analysis
                                .high_frequency_energy_ratio
                            )
                          : "—"}
                      </strong>
                    </div>

                    <div className="noise-frequency-summary-assessment">
                      <span>Overall Assessment</span>

                      <strong>
                        {result.noise_frequency_analysis
                          ?.frequency_characteristic ||
                          "No assessment available"}
                      </strong>
                    </div>
                  </div>
                </div>

                <div className="noise-frequency-reference-note">
                  <span>ⓘ</span>

                  <p>
                    Frequency analysis helps identify unnatural patterns,
                    while noise analysis detects anomalies that may indicate
                    tampering or compression artifacts.
                  </p>
                </div>
              </section>

              <section className="dashboard-card forensic-workbench-card">
                <div className="panel-header">
                  <div>
                    <div className="card-eyebrow">FORENSIC WORKBENCH</div>

                    <h2>Available Analytical Modules</h2>
                  </div>
                </div>

                <div className="planned-evidence-grid">
                  <div className="forensic-module-card">
                    <div className="module-status module-active">ACTIVE</div>
                    <div className="module-icon">◉</div>

                    <h3>Face Detection</h3>

                    <p>
                      Detect and isolate facial regions within submitted
                      evidence.
                    </p>
                  </div>

                  <div className="forensic-module-card">
                    <div className="module-status module-active">ACTIVE</div>
                    <div className="module-icon">◆</div>

                    <h3>Deepfake Classification</h3>

                    <p>
                      Evaluate detected facial imagery using the EfficientNet
                      classification model.
                    </p>
                  </div>

                  <div className="forensic-module-card">
                    <div className="module-status module-active">ACTIVE</div>
                    <div className="module-icon">▣</div>

                    <h3>Metadata / EXIF Analysis</h3>

                    <p>
                      Inspect available image metadata and embedded properties
                      returned by the forensic engine.
                    </p>
                  </div>

                  <div className="forensic-module-card">
                    <div className="module-status module-active">ACTIVE</div>
                    <div className="module-icon">≋</div>

                    <h3>Compression Analysis</h3>

                    <p>
                      Examine compression-related indicators and image
                      integrity signals.
                    </p>
                  </div>

                  <div className="forensic-module-card">
                    <div className="module-status module-active">ACTIVE</div>
                    <div className="module-icon">∿</div>

                    <h3>Noise Analysis</h3>

                    <p>
                      Analyze image noise patterns as an additional forensic
                      signal.
                    </p>
                  </div>

                  <div className="forensic-module-card">
                    <div className="module-status module-active">ACTIVE</div>
                    <div className="module-icon">◌</div>

                    <h3>Frequency Analysis</h3>

                    <p>
                      Inspect frequency-domain characteristics for additional
                      evidence.
                    </p>
                  </div>
                </div>
              </section>

              <section className="dashboard-card evidence-summary-card">
                <div className="panel-header">
                  <div>
                    <div className="card-eyebrow">EVIDENCE SUMMARY</div>
                    <h2>Investigation Record</h2>
                  </div>

                  <span className="asset-status">RECORDED</span>
                </div>

                <div className="report-grid">
                  <div className="report-section">
                    <div className="card-eyebrow">SUBMITTED ASSET</div>

                    <MetadataRow
                      label="Filename"
                      value={selectedFile?.name || "—"}
                    />

                    <MetadataRow
                      label="Content Type"
                      value={
                        result.content_type || selectedFile?.type || "—"
                      }
                    />

                    <MetadataRow
                      label="Dimensions"
                      value={
                        imageWidth && imageHeight
                          ? `${imageWidth} × ${imageHeight}`
                          : "—"
                      }
                    />
                  </div>

                  <div className="report-section">
                    <div className="card-eyebrow">METADATA</div>

                    <MetadataRow
                      label="Format"
                      value={metadata?.format || "—"}
                    />

                    <MetadataRow
                      label="EXIF"
                      value={
                        metadata
                          ? metadata.exif_available
                            ? "AVAILABLE"
                            : "NOT PRESENT"
                          : "—"
                      }
                    />

                    <MetadataRow
                      label="GPS"
                      value={
                        metadata
                          ? metadata.gps_present
                            ? "PRESENT"
                            : "NOT PRESENT"
                          : "—"
                      }
                    />
                  </div>

                  <div className="report-section">
                    <div className="card-eyebrow">ANALYTICAL OUTPUT</div>

                    <MetadataRow label="Verdict" value={getVerdict()} />

                    <MetadataRow
                      label="Confidence"
                      value={formatPercent(modelConfidence)}
                    />

                    <MetadataRow
                      label="Faces"
                      value={String(facesDetected)}
                    />
                  </div>

                  <div className="report-section">
                    <div className="card-eyebrow">ENGINE</div>

                    <MetadataRow
                      label="Model"
                      value={modelName || "EfficientNet-B0"}
                    />

                    <MetadataRow
                      label="Version"
                      value={modelVersion || "Current build"}
                    />

                    <MetadataRow
                      label="Explainability"
                      value={explainabilityStatus || "Not generated"}
                    />
                  </div>
                </div>
              </section>

              <div className="result-disclaimer">
                <span>ⓘ</span>

                <p>
                  Classification output and metadata observations are
                  analytical findings, not standalone proof of authenticity or
                  manipulation. Metadata can be altered or removed
                  independently of the underlying image content. VERITAS does
                  not present unavailable forensic modules as completed
                  evidence.
                </p>
              </div>
            </>
          )}
        </section>
      )}
    </div>
  );
}

function HistoryPage({
  history,
  formatPercent,
}: {
  history: HistoryItem[];
  formatPercent: (value?: number | null) => string;
}) {
  return (
    <div className="page-section">
      <PageHeader
        eyebrow="CASE ARCHIVE"
        title="Investigation History"
        description="Review persistent analysis records generated by the VERITAS forensic engine."
        icon="◷"
      />

      {history.length === 0 ? (
        <div className="empty-state">
          <div className="empty-state-icon">◷</div>

          <h2>No investigations yet</h2>

          <p>
            Completed media analyses will appear here after you run an
            investigation.
          </p>
        </div>
      ) : (
        <div className="history-table-card">
          <div className="table-header history-table-header">
            <div>
              <div className="card-eyebrow">CASE ARCHIVE</div>

              <h2>Investigation Records</h2>

              <p className="panel-description">
                {history.length} investigation
                {history.length === 1 ? "" : "s"} recorded in this workspace
                session.
              </p>
            </div>

            <div className="history-summary-badge">
              <span className="history-summary-dot" />
              <span>DATABASE SYNCED</span>
            </div>
          </div>

          <div className="history-table-wrapper">
            <table className="history-table">
              <thead>
                <tr>
                  <th>Evidence File</th>
                  <th>Verdict</th>
                  <th>Confidence</th>
                  <th>Faces</th>
                  <th>Processing</th>
                  <th>Timestamp</th>
                </tr>
              </thead>

              <tbody>
                {history.map((item) => (
                  <tr key={item.id}>
                    <td>
                      <div className="table-file">
                        <span className="table-file-icon">◇</span>
                        <span>{item.filename}</span>
                      </div>
                    </td>

                    <td>
                      <span
                        className={`table-verdict ${
                          item.verdict === "FAKE"
                            ? "table-verdict-fake"
                            : item.verdict === "REAL"
                              ? "table-verdict-real"
                              : ""
                        }`}
                      >
                        {item.verdict}
                      </span>
                    </td>

                    <td>{formatPercent(item.confidence)}</td>

                    <td>{item.faces}</td>

                    <td>
                      {formatSafeNumber(item.processingTime, 0)} ms
                    </td>

                    <td>{item.timestamp}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function ReportsPage({
  history,
  result,
}: {
  history: HistoryItem[];
  result: AnalysisResult | null;
}) {
  const fakeCount = history.filter((item) => item.verdict === "FAKE").length;
  const realCount = history.filter((item) => item.verdict === "REAL").length;

  const reportAvailable =
    result?.report?.available === true &&
    Boolean(result.report.filename);

  async function openReport() {
    const filename = result?.report?.filename;

    if (!filename) {
      window.alert(
        "No generated forensic report is available yet. Run an analysis first."
      );
      return;
    }

    const reportUrl =
      `${API_URL}/reports/${encodeURIComponent(filename)}`;

    try {
      const response = await fetch(reportUrl);

      if (!response.ok) {
        throw new Error(`Report endpoint returned ${response.status}.`);
      }

      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);

      window.open(blobUrl, "_blank", "noopener,noreferrer");

      window.setTimeout(() => URL.revokeObjectURL(blobUrl), 60000);
    } catch (error) {
      console.error("Forensic report download error:", error);

      window.alert(
        "The forensic report could not be opened. Make sure the backend is running and the report file is available."
      );
    }
  }

  return (
    <div className="page-section">
      <PageHeader
        eyebrow="FORENSIC DOCUMENTATION"
        title="Reports"
        description="Generated investigation reports and evidence summaries from the VERITAS forensic engine."
        icon="▤"
      />

      <div className="report-overview-grid">
        <MetricCard
          label="TOTAL CASES"
          value={String(history.length)}
          detail="Persistent analyses stored in the database"
          icon="▣"
          status="neutral"
        />

        <MetricCard
          label="REAL CASES"
          value={String(realCount)}
          detail="Model classifications"
          icon="✓"
          status="good"
        />

        <MetricCard
          label="FAKE CASES"
          value={String(fakeCount)}
          detail="Model classifications"
          icon="!"
          status="warning"
        />
      </div>

      <div className="report-builder-card">
        <div className="report-icon">▥</div>

        <div className="report-builder-copy">
          <div className="card-eyebrow">REPORT GENERATION</div>

          <h2>Forensic Report Builder</h2>

          <p>
            VERITAS generates a structured forensic report after a completed
            media analysis. The report is linked to the latest investigation
            and contains the evidence summary produced by the backend forensic
            engine.
          </p>

          {reportAvailable ? (
            <>
              <div className="panel-header">
                <div>
                  <span className="card-eyebrow">LATEST REPORT</span>
                  <strong>{result?.report?.filename}</strong>
                </div>

                <button
                  type="button"
                  className="primary-button"
                  onClick={() => void openReport()}
                >
                  <span>Open PDF Report</span>
                  <span>→</span>
                </button>
              </div>

              <span className="history-summary-badge">REPORT READY</span>
            </>
          ) : (
            <>
              <span className="planned-badge">
                {result
                  ? "REPORT NOT AVAILABLE"
                  : "RUN AN ANALYSIS TO GENERATE A REPORT"}
              </span>

              <p>
                {result
                  ? "The latest analysis did not return a downloadable report file."
                  : "Upload an image from New Investigation. Once the analysis completes, the generated report will appear here."}
              </p>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

function ModulesPage({
  activeModules,
  plannedModules,
}: {
  activeModules: number;
  plannedModules: number;
}) {
  return (
    <div className="page-section">
      <PageHeader
        eyebrow="FORENSIC CAPABILITIES"
        title="Forensic Modules"
        description="VERITAS analysis engines, evidence indicators, and future forensic capabilities."
        icon="◈"
      />

      <div className="module-summary">
        <div>
          <strong>{activeModules}</strong>
          <span>ACTIVE MODULES</span>
        </div>

        <div>
          <strong>{plannedModules}</strong>
          <span>PLANNED MODULES</span>
        </div>

        <div>
          <strong>{activeModules + plannedModules}</strong>
          <span>TOTAL CAPABILITIES</span>
        </div>
      </div>

      <div className="module-grid">
        {modules.map((module) => (
          <div
            className={`module-card ${
              module.status === "ACTIVE"
                ? "module-active"
                : "module-planned"
            }`}
            key={module.number}
          >
            <div className="module-top">
              <span className="module-number">{module.number}</span>

              <span
                className={`module-status ${
                  module.status === "ACTIVE"
                    ? "module-status-active"
                    : "module-status-planned"
                }`}
              >
                {module.status}
              </span>
            </div>

            <div className="module-icon">{module.icon}</div>

            <h3>{module.title}</h3>

            <p>{module.description}</p>

            <div className="module-footer">
              <span>
                {module.status === "ACTIVE"
                  ? "Available now"
                  : "Roadmap capability"}
              </span>

              <span>→</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function SystemPage({
  backendOnline,
  checkBackend,
}: {
  backendOnline: boolean;
  checkBackend: () => Promise<void>;
}) {
  return (
    <div className="page-section">
      <PageHeader
        eyebrow="SYSTEM MONITORING"
        title="System Status"
        description="Current VERITAS application and forensic engine readiness."
        icon="⚙"
      />

      <div className="system-status-card">
        <div
          className={`system-status-large ${
            backendOnline
              ? "system-status-good"
              : "system-status-warning"
          }`}
        >
          <div className="system-status-icon">
            {backendOnline ? "✓" : "!"}
          </div>

          <div>
            <div className="card-eyebrow">FASTAPI FORENSIC ENGINE</div>

            <h2>
              {backendOnline ? "Operational" : "Connection unavailable"}
            </h2>

            <p>
              {backendOnline
                ? "The VERITAS frontend can communicate with the media analysis service."
                : "The frontend cannot currently reach the analysis service at 127.0.0.1:8000."}
            </p>
          </div>

          <button
            className="primary-button"
            onClick={() => void checkBackend()}
          >
            Recheck Service
          </button>
        </div>

        <div className="system-components">
          <SystemComponent
            title="Frontend"
            value="React + TypeScript"
            status="READY"
          />

          <SystemComponent
            title="API"
            value="FastAPI / Port 8000"
            status={backendOnline ? "ONLINE" : "OFFLINE"}
          />

          <SystemComponent
            title="Classifier"
            value="EfficientNet-B0"
            status="LOADED"
          />

          <SystemComponent
            title="Face Detector"
            value="OpenCV Haar Cascade"
            status="LOADED"
          />

          <SystemComponent
            title="Database"
            value="SQLite / SQLAlchemy"
            status="CONFIGURED"
          />

          <SystemComponent
            title="Explainability"
            value="Grad-CAM"
            status="ACTIVE"
          />
        </div>
      </div>

      <div className="system-note">
        <span>ⓘ</span>

        <p>
          A loaded model does not automatically mean that forensic performance
          is validated. Proper evaluation, including cross-dataset testing,
          remains part of the VERITAS validation workflow.
        </p>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  detail,
  icon,
  status,
}: {
  label: string;
  value: string;
  detail: string;
  icon: string;
  status: "good" | "warning" | "neutral";
}) {
  return (
    <div className="metric-card">
      <div className="metric-top">
        <span className="metric-label">{label}</span>

        <span className={`metric-icon metric-icon-${status}`}>
          {icon}
        </span>
      </div>

      <div className="metric-value">{value}</div>

      <div className="metric-detail">{detail}</div>
    </div>
  );
}

function PipelineItem({
  number,
  title,
  description,
  status,
}: {
  number: string;
  title: string;
  description: string;
  status: "ready" | "planned";
}) {
  return (
    <div className="pipeline-item">
      <div
        className={`pipeline-marker ${
          status === "ready"
            ? "pipeline-marker-ready"
            : "pipeline-marker-planned"
        }`}
      >
        {status === "ready" ? "✓" : "—"}
      </div>

      <div className="pipeline-number">{number}</div>

      <div className="pipeline-content">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>

      <div
        className={`pipeline-state ${
          status === "ready"
            ? "pipeline-state-ready"
            : "pipeline-state-planned"
        }`}
      >
        {status === "ready" ? "READY" : "PLANNED"}
      </div>
    </div>
  );
}

function QuickAction({
  icon,
  title,
  description,
  onClick,
}: {
  icon: string;
  title: string;
  description: string;
  onClick: () => void;
}) {
  return (
    <button className="quick-action" onClick={onClick}>
      <span className="quick-action-icon">{icon}</span>

      <span className="quick-action-copy">
        <strong>{title}</strong>
        <small>{description}</small>
      </span>

      <span className="quick-action-arrow">→</span>
    </button>
  );
}

function MetadataRow({
  label,
  value,
}: {
  label: string;
  value: ReactNode;
}) {
  return (
    <div className="metadata-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ResultMetric({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="result-metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SystemComponent({
  title,
  value,
  status,
}: {
  title: string;
  value: string;
  status: string;
}) {
  const positiveStatuses = [
    "READY",
    "ONLINE",
    "LOADED",
    "CONFIGURED",
    "ACTIVE",
  ];

  const isPositive = positiveStatuses.includes(status);

  return (
    <div className="system-component">
      <div>
        <strong>{title}</strong>
        <span>{value}</span>
      </div>

      <span
        className={`system-component-status ${
          isPositive
            ? "component-status-good"
            : "component-status-planned"
        }`}
      >
        {status}
      </span>
    </div>
  );
}

export default App;
