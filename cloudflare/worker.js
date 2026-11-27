/**
 * AI Video Factory — Free-Tier Cloudflare Worker Orchestrator
 *
 * Strictly lightweight API + D1 state tracking + Cron Trigger.
 * Never executes ML models or FFmpeg inside the Worker (10 ms CPU limit).
 * Set secret via: `wrangler secret put API_KEY`
 */

function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function isAuthorized(request, env) {
  if (!env.API_KEY) return true; // Local dev fallback if unset
  const auth = request.headers.get("Authorization") || "";
  return auth === `Bearer ${env.API_KEY}`;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    if (path === "/health") {
      return jsonResponse({ ok: true, tier: "cloudflare-free" });
    }

    if (!isAuthorized(request, env)) {
      return jsonResponse({ error: "Unauthorized" }, 401);
    }

    // POST /jobs — Create a new queued production job
    if (request.method === "POST" && path === "/jobs") {
      const body = await request.json();
      if (!body.topic) {
        return jsonResponse({ error: "topic is required" }, 400);
      }
      const jobId = body.job_id || `job_${Date.now()}`;
      const now = new Date().toISOString();

      await env.DB.prepare(
        `INSERT INTO jobs (job_id, topic, audience, tone, style, target_duration_sec, num_shorts, state, created_at, updated_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, 'QUEUED', ?, ?)`
      )
        .bind(
          jobId,
          body.topic,
          body.audience || "curious general audience",
          body.tone || "engaging, clear, documentary",
          body.style || "cinematic, clean editorial visuals",
          body.target_duration_sec || 600,
          body.num_shorts || 3,
          now,
          now
        )
        .run();

      return jsonResponse({ job_id: jobId, state: "QUEUED" }, 201);
    }

    // GET /jobs/next — Kaggle notebook polls for oldest QUEUED or PAUSED job
    if (request.method === "GET" && path === "/jobs/next") {
      const row = await env.DB.prepare(
        `SELECT * FROM jobs WHERE state IN ('QUEUED', 'PAUSED') ORDER BY created_at ASC LIMIT 1`
      ).first();
      return jsonResponse({ job: row || null });
    }

    // PATCH /jobs/:id — Update state & manifest from Kaggle runner
    if (request.method === "PATCH" && path.startsWith("/jobs/")) {
      const jobId = path.replace("/jobs/", "");
      const body = await request.json();
      const now = new Date().toISOString();

      await env.DB.prepare(
        `UPDATE jobs SET state = COALESCE(?, state), manifest_json = COALESCE(?, manifest_json), error_message = COALESCE(?, error_message), updated_at = ? WHERE job_id = ?`
      )
        .bind(
          body.state || null,
          body.manifest ? JSON.stringify(body.manifest) : null,
          body.error || null,
          now,
          jobId
        )
        .run();

      return jsonResponse({ job_id: jobId, updated: true });
    }

    return jsonResponse({ error: "Not found" }, 404);
  },

  async scheduled(_event, env) {
    // Lightweight housekeeping: mark stale INITIALISING jobs (>13h) as PAUSED so they can resume
    const cutoff = new Date(Date.now() - 13 * 3600 * 1000).toISOString();
    const now = new Date().toISOString();
    await env.DB.prepare(
      `UPDATE jobs SET state = 'PAUSED', updated_at = ? WHERE state NOT IN ('QUEUED', 'COMPLETED', 'FAILED', 'PAUSED') AND updated_at < ?`
    )
      .bind(now, cutoff)
      .run();
  },
};
