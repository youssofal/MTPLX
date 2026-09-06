#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { gzipSync } from "node:zlib";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const dashboard = join(root, "dashboard");
const output = join(root, "mtplx", "dashboard", "_static");
const iterations = Number.parseInt(process.argv[2] ?? "3", 10);

if (!Number.isInteger(iterations) || iterations < 1) {
  throw new Error("iterations must be a positive integer");
}

const buildMs = [];
for (let iteration = 0; iteration < iterations; iteration += 1) {
  const start = process.hrtime.bigint();
  const result = spawnSync("npm", ["run", "build"], {
    cwd: dashboard,
    encoding: "utf8",
  });
  if (result.status !== 0) {
    process.stderr.write(result.stdout);
    process.stderr.write(result.stderr);
    process.exit(result.status ?? 1);
  }
  buildMs.push(Number(process.hrtime.bigint() - start) / 1e6);
}

function filesUnder(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? filesUnder(path) : [path];
  });
}

const files = filesUnder(output)
  .sort()
  .map((path) => {
    const bytes = statSync(path).size;
    const gzipBytes = gzipSync(readFileSync(path)).length;
    return { path: relative(output, path), bytes, gzip_bytes: gzipBytes };
  });
const sortedBuildMs = [...buildMs].sort((a, b) => a - b);

console.log(
  JSON.stringify(
    {
      iterations,
      build_ms: buildMs,
      median_build_ms: sortedBuildMs[Math.floor(sortedBuildMs.length / 2)],
      total_bytes: files.reduce((total, file) => total + file.bytes, 0),
      total_gzip_bytes: files.reduce((total, file) => total + file.gzip_bytes, 0),
      files,
    },
    null,
    2,
  ),
);
