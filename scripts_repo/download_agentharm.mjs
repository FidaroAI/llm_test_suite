#!/usr/bin/env node
// One-off downloader for the ai-safety-institute/AgentHarm dataset
// (chat config, test_public split).
//
// Pulls every row from the Hugging Face datasets-server rows API and writes the
// raw rows (untransformed) to data/agentharm.json. All shaping into promptfoo
// test cases happens later in tests/agentharm_refusal_gen.py.
//
// AgentHarm is a gated dataset; if the rows API returns 401/403 you must accept
// the dataset terms at https://huggingface.co/datasets/ai-safety-institute/AgentHarm
// (and optionally export HF_TOKEN, which this script forwards if present).
//
// Usage: npm run dataset:agentharm

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const DATASET = "ai-safety-institute/AgentHarm";
const CONFIG = "chat";
const SPLIT = "test_public";
const PAGE_SIZE = 100; // datasets-server caps a single request at 100 rows
const ROWS_URL = "https://datasets-server.huggingface.co/rows";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outPath = resolve(repoRoot, "data", "agentharm.json");

const headers = process.env.HF_TOKEN
  ? { Authorization: `Bearer ${process.env.HF_TOKEN}` }
  : {};

async function fetchPage(offset) {
  const url = `${ROWS_URL}?dataset=${encodeURIComponent(DATASET)}&config=${CONFIG}&split=${SPLIT}&offset=${offset}&length=${PAGE_SIZE}`;
  const res = await fetch(url, { headers });
  if (res.status === 401 || res.status === 403) {
    throw new Error(
      `HF rows API ${res.status}: AgentHarm is gated. Accept the terms at ` +
        "https://huggingface.co/datasets/ai-safety-institute/AgentHarm and/or " +
        "export HF_TOKEN before running this script.",
    );
  }
  if (!res.ok) {
    throw new Error(`HF rows API ${res.status} ${res.statusText} for offset ${offset}`);
  }
  return res.json();
}

async function main() {
  const rows = [];
  let offset = 0;
  let total = Infinity;

  while (offset < total) {
    const page = await fetchPage(offset);
    total = page.num_rows_total ?? page.rows.length;
    // Each entry is { row_idx, row: {...} }; we keep just the data object.
    for (const entry of page.rows) rows.push(entry.row);
    if (page.rows.length === 0) break;
    offset += page.rows.length;
    process.stdout.write(`\rdownloaded ${rows.length}/${total} rows`);
  }
  process.stdout.write("\n");

  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, JSON.stringify(rows, null, 2));
  console.log(`wrote ${rows.length} rows to ${outPath}`);
}

main().catch((err) => {
  console.error(err.message ?? err);
  process.exit(1);
});
