-- Cloudflare D1 Schema (Free Tier: 5 GB total storage, 500 MB max per DB)
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  topic TEXT NOT NULL,
  audience TEXT DEFAULT 'curious general audience',
  tone TEXT DEFAULT 'engaging, clear, documentary',
  style TEXT DEFAULT 'cinematic, clean editorial visuals',
  target_duration_sec INTEGER DEFAULT 600,
  num_shorts INTEGER DEFAULT 3,
  state TEXT NOT NULL DEFAULT 'QUEUED',
  manifest_json TEXT,
  error_message TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
