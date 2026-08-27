import {expect, test} from "@playwright/test";

const API_KEY = "service-key-0123456789abcdef";

async function connect(page) {
  await page.goto("/app");
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.getByLabel("API Key").fill(API_KEY);
  await page.getByRole("button", {name: "安全连接"}).click();
  await expect(page.getByText("连接成功，密钥仅保存在当前标签页。")).toBeVisible();
}

async function createReadyKnowledgeBase(page) {
  await page.getByRole("button", {name: "新建知识库"}).click();
  await page.getByLabel("知识库名称").fill("浏览器端资料");
  await page.locator("#documents").setInputFiles({
    name: "evidence.md",
    mimeType: "text/markdown",
    buffer: Buffer.from("evidence"),
  });
  await expect(page.getByText("evidence.md")).toBeVisible();
  await page.getByRole("button", {name: "开始建立索引"}).click();

  await expect(page.getByText("知识库已就绪，可以开始提问。")).toBeVisible();
  await expect(page.locator("#active-status")).toHaveText("READY");
  await expect(page.getByRole("heading", {name: "Engineering guide"})).toBeVisible();
}

test("a user can create, query, inspect, clear, and delete a knowledge base", async ({page}) => {
  await connect(page);
  await createReadyKnowledgeBase(page);

  await page.getByLabel("输入问题").fill("RAG 如何工作？");
  await page.getByRole("button", {name: "发送问题"}).click();
  await expect(page.locator(".message.assistant")).toContainText(
    "RAG retrieves evidence before generation.",
  );
  await expect(page.locator(".citation")).toContainText("guide.md");

  await page.getByRole("button", {name: "资料详情"}).click();
  await expect(page.getByRole("dialog")).toContainText("SHA-256");
  await page.getByRole("button", {name: "关闭"}).click();

  await page.getByRole("button", {name: "清空对话"}).click();
  await expect(page.locator(".message")).toHaveCount(0);

  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", {name: "删除知识库"}).click();
  await expect(page.getByText("还没有知识库。上传资料后会显示在这里。")).toBeVisible();
});

test("external options require explicit consent and keep the local default", async ({page}) => {
  await connect(page);
  await createReadyKnowledgeBase(page);
  const cloud = page.locator("#allow-cloud");
  await expect(cloud).not.toBeChecked();
  await page.getByText("云端生成", {exact: true}).click();
  await expect(page.getByRole("dialog")).toContainText("开启云端生成？");
  await expect(page.getByRole("dialog")).toContainText("不发送：本地访问密钥");
  await page.getByRole("button", {name: "保持本地"}).click();
  await expect(cloud).not.toBeChecked();
  await expect(page.locator("#privacy-note")).toHaveText("内容不会发送到外部服务");
});

test("the workbench keeps authentication and ingestion failures actionable", async ({page}) => {
  await page.goto("/app");
  await page.getByLabel("API Key").fill("wrong-key-0123456789");
  await page.getByRole("button", {name: "安全连接"}).click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await expect(page.locator("#key-error")).toHaveText("密钥验证失败，请重新复制。");

  await page.getByLabel("API Key").fill(API_KEY);
  await page.getByRole("button", {name: "安全连接"}).click();
  await page.getByRole("button", {name: "新建知识库"}).click();
  await page.getByLabel("知识库名称").fill("故障资料");
  await page.locator("#documents").setInputFiles({
    name: "unavailable.md",
    mimeType: "text/markdown",
    buffer: Buffer.from("evidence"),
  });
  await page.getByRole("button", {name: "开始建立索引"}).click();
  await expect(page.locator("#create-error")).toHaveText("The service is temporarily unavailable.");
  await expect(page.locator("#task-banner")).toHaveClass(/hidden/);
});

test("legacy session keys are promoted without dropping the existing session", async ({page}) => {
  await page.addInitScript((apiKey) => {
    window.sessionStorage.setItem("rag-studio-api-key", apiKey);
  }, API_KEY);
  const authenticatedList = page.waitForResponse((response) => (
    response.url().endsWith("/v1/knowledge-bases?limit=100") && response.status() === 200
  ));

  await page.goto("/app");
  await authenticatedList;
  await expect.poll(() => page.evaluate(() => window.sessionStorage.getItem("rag-platform-api-key")))
    .toBe(API_KEY);
  await expect.poll(() => page.evaluate(() => window.sessionStorage.getItem("rag-studio-api-key")))
    .toBe(API_KEY);
});
