import fs from "node:fs/promises";
import { PDFParse } from "pdf-parse";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";
import { OpenAIEmbeddings, ChatOpenAI } from "@langchain/openai";

export const DEFAULT_CHUNK_SIZE = 1200;
export const DEFAULT_CHUNK_OVERLAP = 200;
export const DEFAULT_TOP_K = 4;

export async function extractPdfText(pdfPath) {
  const fileBuffer = await fs.readFile(pdfPath);
  const parser = new PDFParse({ data: fileBuffer });
  const parsed = await parser.getText();
  await parser.destroy();

  if (!parsed.text || !parsed.text.trim()) {
    throw new Error("O PDF não contém texto legível para indexação.");
  }

  return parsed.text;
}

export async function splitTextIntoChunks(text, chunkSize = DEFAULT_CHUNK_SIZE, chunkOverlap = DEFAULT_CHUNK_OVERLAP) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize,
    chunkOverlap,
  });

  return splitter.createDocuments([text]);
}

export async function buildVectorStoreFromPdf(pdfPath) {
  const text = await extractPdfText(pdfPath);
  const chunks = await splitTextIntoChunks(text);

  const embeddings = new OpenAIEmbeddings({
    model: "text-embedding-3-small",
  });

  const vectorStore = await MemoryVectorStore.fromDocuments(chunks, embeddings);
  return vectorStore;
}

export async function answerQuestion(vectorStore, question, topK = DEFAULT_TOP_K) {
  const relevantChunks = await vectorStore.similaritySearch(question, topK);

  const context = relevantChunks
    .map((doc, index) => `[Trecho ${index + 1}]\n${doc.pageContent}`)
    .join("\n\n");

  const prompt = [
    "Você é um assistente de RAG.",
    "Use apenas o contexto abaixo para responder.",
    "Se a informação não estiver presente, diga que não encontrou no PDF.",
    "",
    "Contexto:",
    context || "(sem contexto relevante)",
    "",
    `Pergunta: ${question}`,
  ].join("\n");

  const model = new ChatOpenAI({
    model: "gpt-4.1-mini",
    temperature: 0,
  });

  const response = await model.invoke(prompt);
  if (typeof response.content === "string") {
    return response.content;
  }

  if (Array.isArray(response.content)) {
    return response.content.map((part) => (typeof part === "string" ? part : part.text || "")).join(" ").trim();
  }

  return "Não foi possível gerar uma resposta.";
}
