import {spawn, spawnSync} from "node:child_process";
import {resolve} from "node:path";

const port = 4173;
const baseUrl = `http://127.0.0.1:${port}`;
const python = process.env.PYTHON || (process.platform === "win32"
  ? resolve(".venv", "Scripts", "python.exe")
  : "python");
const playwrightCli = resolve("node_modules", "@playwright", "test", "cli.js");

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
  python,
  ["-m", "uvicorn", "tests.browser_e2e_app:app", "--host", "127.0.0.1", "--port", String(port)],
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
