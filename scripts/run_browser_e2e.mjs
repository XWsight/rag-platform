import {spawn, spawnSync} from "node:child_process";
import {existsSync} from "node:fs";
import {resolve} from "node:path";

const port = 4173;
const baseUrl = `http://127.0.0.1:${port}`;
const playwrightCli = resolve("node_modules", "@playwright", "test", "cli.js");

function selectPython() {
  const configured = process.env.PYTHON?.trim();
  if (configured) return {command: configured, prefixArguments: []};

  const virtualEnvironmentPython = process.platform === "win32"
    ? resolve(".venv", "Scripts", "python.exe")
    : resolve(".venv", "bin", "python");
  if (existsSync(virtualEnvironmentPython)) {
    return {command: virtualEnvironmentPython, prefixArguments: []};
  }

  if (process.platform === "win32") {
    return {command: "py", prefixArguments: ["-3.11"]};
  }
  return {command: "python3", prefixArguments: []};
}

function assertSupportedPython(python) {
  const result = spawnSync(
    python.command,
    [...python.prefixArguments, "-c", "import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 13) else 1)"],
    {encoding: "utf8", windowsHide: true},
  );
  if (result.error || result.status !== 0) {
    const detail = [result.error?.message, result.stderr?.trim()].filter(Boolean).join("; ");
    throw new Error(
      `Browser E2E requires Python 3.11 or 3.12. Create .venv or set PYTHON to a supported interpreter.${detail ? ` (${detail})` : ""}`,
    );
  }
}

if (!existsSync(playwrightCli)) {
  throw new Error("Browser test dependencies are missing. Run npm ci before npm run test:browser.");
}

const python = selectPython();
assertSupportedPython(python);

function sleep(milliseconds) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}

async function waitForReady(server, startupError) {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    if (startupError.value) throw startupError.value;
    if (server.exitCode !== null) throw new Error(`browser fixture exited with ${server.exitCode}`);
    try {
      const response = await fetch(`${baseUrl}/health/ready`);
      if (response.ok) return;
    } catch {
      // The local fixture has not bound its socket yet.
    }
    await sleep(100);
  }
  throw new Error("browser fixture did not become ready within 30 seconds");
}

function run(commandName, argumentsList, options) {
  return new Promise((resolveExit, rejectExit) => {
    const child = spawn(commandName, argumentsList, options);
    child.once("error", rejectExit);
    child.once("exit", (code, signal) => resolveExit({code, signal}));
  });
}

function stopProcessTree(server) {
  if (server.exitCode !== null || server.pid === undefined) return;
  if (process.platform === "win32") {
    spawnSync("taskkill", ["/pid", String(server.pid), "/T", "/F"], {stdio: "ignore"});
  } else {
    server.kill("SIGTERM");
  }
}

const server = spawn(
  python.command,
  [...python.prefixArguments, "-m", "uvicorn", "tests.browser_e2e_app:app", "--host", "127.0.0.1", "--port", String(port)],
  {stdio: "inherit", windowsHide: true},
);
const startupError = {value: null};
server.once("error", (error) => { startupError.value = error; });

try {
  await waitForReady(server, startupError);
  const result = await run(process.execPath, [playwrightCli, "test"], {
    stdio: "inherit",
    env: {...process.env, PLAYWRIGHT_BASE_URL: baseUrl},
  });
  if (result.code !== 0) {
    throw new Error(`browser end-to-end tests failed (${result.signal || result.code})`);
  }
} finally {
  stopProcessTree(server);
}
