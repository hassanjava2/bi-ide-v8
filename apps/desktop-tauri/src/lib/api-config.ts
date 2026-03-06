/**
 * api-config.ts — إعدادات API مركزية
 * 
 * الـ Brain API يشتغل على RTX 5090 (عبر Tailscale)
 * الـ IDE يشتغل على Mac (localhost)
 * 
 * يكتشف أوتوماتيكياً: إذا RTX متصل يستخدمه، وإلا localhost
 */

// الأولوية: RTX 5090 عبر Tailscale → localhost fallback
const RTX_5090_URL = "http://100.104.35.44:8400";
const LOCAL_URL = "http://localhost:8400";

// Cache result
let _cachedUrl: string | null = null;
let _lastCheck = 0;

/**
 * يكتشف أفضل URL للـ API
 * يفحص RTX أولاً → إذا ما رد يستخدم localhost
 */
export async function detectApiUrl(): Promise<string> {
    const now = Date.now();
    // Re-check every 30 seconds
    if (_cachedUrl && now - _lastCheck < 30000) {
        return _cachedUrl;
    }

    try {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 3000);
        const res = await fetch(`${RTX_5090_URL}/api/status`, {
            signal: controller.signal,
        });
        clearTimeout(timeout);

        if (res.ok) {
            _cachedUrl = RTX_5090_URL;
            _lastCheck = now;
            console.log("🧠 Brain API: RTX 5090 connected", RTX_5090_URL);
            return RTX_5090_URL;
        }
    } catch {
        // RTX not reachable
    }

    _cachedUrl = LOCAL_URL;
    _lastCheck = now;
    console.log("🧠 Brain API: using localhost", LOCAL_URL);
    return LOCAL_URL;
}

/**
 * يرجع URL الحالي (sync — يستخدم الـ cached)
 */
export function getApiUrl(): string {
    return _cachedUrl || RTX_5090_URL;
}

/**
 * Fetch wrapper — يستخدم الـ API URL الصحيح
 */
export async function apiFetch(path: string, options?: RequestInit): Promise<Response> {
    const base = await detectApiUrl();
    return fetch(`${base}${path}`, options);
}

/**
 * POST JSON helper
 */
export async function apiPost(path: string, body: any): Promise<any> {
    const res = await apiFetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
    return res.json();
}

/**
 * GET JSON helper
 */
export async function apiGet(path: string): Promise<any> {
    const res = await apiFetch(path);
    return res.json();
}
