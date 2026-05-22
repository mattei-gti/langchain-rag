import test from "node:test";
import assert from "node:assert/strict";
import { splitTextIntoChunks } from "../src/rag.js";

test("splitTextIntoChunks cria múltiplos chunks para textos longos", async () => {
  const longText = "A".repeat(1800);
  const docs = await splitTextIntoChunks(longText, 700, 100);

  assert.ok(docs.length >= 2);
  assert.ok(docs.every((doc) => doc.pageContent.length <= 700));
});
