import "dotenv/config";
import path from "node:path";
import process from "node:process";
import readline from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { buildVectorStoreFromPdf, answerQuestion } from "./rag.js";

async function main() {
  const pdfPathArg = process.argv[2];

  if (!pdfPathArg) {
    console.error("Uso: npm start -- <caminho-do-arquivo.pdf>");
    process.exitCode = 1;
    return;
  }

  if (!process.env.OPENAI_API_KEY) {
    console.error("Defina OPENAI_API_KEY no ambiente (ou em .env).\n");
    process.exitCode = 1;
    return;
  }

  const pdfPath = path.resolve(process.cwd(), pdfPathArg);

  console.log("📄 Carregando e indexando PDF...");
  const vectorStore = await buildVectorStoreFromPdf(pdfPath);
  console.log("✅ PDF indexado com sucesso.");
  console.log("💬 Faça perguntas (digite 'sair' para encerrar).\n");

  const rl = readline.createInterface({ input, output });

  try {
    while (true) {
      const question = (await rl.question("Você: ")).trim();

      if (!question) {
        continue;
      }

      if (["sair", "exit", "quit"].includes(question.toLowerCase())) {
        console.log("Até mais!");
        break;
      }

      const answer = await answerQuestion(vectorStore, question);
      console.log(`\nAssistente: ${answer}\n`);
    }
  } finally {
    rl.close();
  }
}

main().catch((error) => {
  console.error("❌ Erro ao executar o assistente:", error.message);
  process.exitCode = 1;
});
