#!/usr/bin/env node
// One-off downloader for the ScaleAI/researchrubrics dataset.
//
// Pulls every row from the Hugging Face datasets-server rows API and writes the
// raw rows (untransformed) to data/researchrubrics.json. All shaping into
// promptfoo test cases happens later in tests/research_rubrics_gen.py.
//
// Usage: npm run dataset:researchrubrics

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const DATASET = "ScaleAI/researchrubrics";
const CONFIG = "default";
const SPLIT = "train";
const PAGE_SIZE = 100; // datasets-server caps a single request at 100 rows
const ROWS_URL = "https://datasets-server.huggingface.co/rows";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outPath = resolve(repoRoot, "data", "researchrubrics.json");

async function fetchPage(offset) {
  const url = `${ROWS_URL}?dataset=${encodeURIComponent(DATASET)}&config=${CONFIG}&split=${SPLIT}&offset=${offset}&length=${PAGE_SIZE}`;
  const res = await fetch(url);
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
