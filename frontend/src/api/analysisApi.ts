const API_BASE_URL = "http://127.0.0.1:8000";

/* =========================================================
   ANALYSIS RESULT
========================================================= */

export interface AnalysisResult {
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
  heatmap_base64?: string;

  processing_time_ms?: number;

  trained_for_deepfake_detection?: boolean;
  device?: string;
  face_detection_status?: string;

  notes?: string;

  [key: string]: unknown;
}

/* =========================================================
   HEALTH RESULT
========================================================= */

export interface HealthResult {
  status: string;

  model_loaded?: boolean;
  face_detector_loaded?: boolean;

  device?: string;

  trained_for_deepfake_detection?: boolean;

  [key: string]: unknown;
}

/* =========================================================
   API ERROR
========================================================= */

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  return "An unknown error occurred while communicating with the analysis service.";
}

/* =========================================================
   CHECK API HEALTH
========================================================= */

export async function checkHealth(): Promise<HealthResult> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    });

    let data: HealthResult | null = null;

    try {
      data = (await response.json()) as HealthResult;
    } catch {
      data = null;
    }

    if (!response.ok) {
      throw new Error(
        data && typeof data.detail === "string"
          ? data.detail
          : `Health check failed with status ${response.status}.`
      );
    }

    if (!data) {
      throw new Error("The backend returned an empty health response.");
    }

    return data;
  } catch (error) {
    throw new Error(`Backend connection failed: ${getErrorMessage(error)}`);
  }
}

/* =========================================================
   ANALYZE IMAGE
========================================================= */

export async function analyzeImage(
  file: File
): Promise<AnalysisResult> {
  if (!file) {
    throw new Error("No image file was selected.");
  }

  if (!file.type.startsWith("image/")) {
    throw new Error("Please select a valid image file.");
  }

  const formData = new FormData();

  formData.append("file", file, file.name);

  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: "POST",
      body: formData,
    });

    let data: AnalysisResult | null = null;

    try {
      data = (await response.json()) as AnalysisResult;
    } catch {
      data = null;
    }

    if (!response.ok) {
      let serverMessage = `Analysis failed with status ${response.status}.`;

      if (
        data &&
        typeof data.detail === "string" &&
        data.detail.trim()
      ) {
        serverMessage = data.detail;
      }

      throw new Error(serverMessage);
    }

    if (!data) {
      throw new Error(
        "The analysis service returned an empty response."
      );
    }

    return data;
  } catch (error) {
    const message = getErrorMessage(error);

    /*
     * A browser-level "Failed to fetch" normally means that
     * the request could not reach the backend at all.
     */
    if (
      message.toLowerCase().includes("failed to fetch") ||
      message.toLowerCase().includes("networkerror") ||
      message.toLowerCase().includes("load failed")
    ) {
      throw new Error(
        "Unable to connect to the VERITAS analysis service. " +
        "Make sure the FastAPI backend is running at " +
        `${API_BASE_URL}.`
      );
    }

    throw new Error(message);
  }
}

/* =========================================================
   DEFAULT EXPORT
========================================================= */

export default {
  analyzeImage,
  checkHealth,
};