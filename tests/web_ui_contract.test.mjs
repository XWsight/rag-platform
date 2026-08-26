import assert from "node:assert/strict";
import test from "node:test";
import {readFile} from "node:fs/promises";
import {fileURLToPath} from "node:url";

const asset = fileURLToPath(new URL("../rag_system/web_ui/assets/app.js", import.meta.url));
const source = await readFile(asset, "utf8");

test("browser client keeps the documented same-origin API contract", () => {
  for (const endpoint of [
    '"/app/config"',
    '"/health/ready"',
    '"/v1/knowledge-bases"',
    '"/v1/answers"',
  ]) {
    assert.match(source, new RegExp(endpoint.replaceAll("/", "\\/")));
  }
  assert.match(source, /`\/v1\/jobs\/\$\{encodeURIComponent\(jobId\)\}`/);
  assert.match(source, /X-API-Key/);
});

test("untrusted service values are rendered as text, not HTML", () => {
  assert.doesNotMatch(source, /\.innerHTML\s*=/);
  assert.match(source, /\.textContent\s*=/);
  assert.match(source, /encodeURIComponent\(active\.id\)/);
  assert.match(source, /encodeURIComponent\(state\.sessionId\)/);
});
