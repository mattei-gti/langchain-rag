const fs = require('fs/promises');
const path = require('path');
const readline = require('readline/promises');

const dotenv = require('dotenv');
const pdfParse = require('pdf-parse');
const { Document } = require('@langchain/core/documents');
const { RecursiveCharacterTextSplitter } = require('langchain/text_splitter');
const { MemoryVectorStore } = require('langchain/vectorstores/memory');
const { ChatOpenAI, OpenAIEmbeddings } = require('@langchain/openai');

dotenv.config();

function getCliPdfPath() {
  const inputPath = process.argv[2];
  if (!inputPath) {
    throw new Error('Informe o caminho do PDF. Exemplo: npm start -- ./documento.pdf');
  }

  return path.resolve(process.cwd(), inputPath);
}

function normalizeAiResponse(response) {
  if (!response?.content) {
    return 'Não foi possível gerar uma resposta.';
  }

  if (typeof response.content === 'string') {
    return response.content;
  }

  if (Array.isArray(response.content)) {
    return response.content
      .map((item) => {
        if (typeof item === 'string') {
          return item;
        }

        if (item && typeof item.text === 'string') {
          return item.text;
        }

        return '';
      })
      .join('')
      .trim();
  }

  return String(response.content);
}

async function createVectorStoreFromPdf(pdfPath) {
  const buffer = await fs.readFile(pdfPath);
  const parsedPdf = await pdfParse(buffer);

  if (!parsedPdf?.text?.trim()) {
    throw new Error('Não foi possível extrair texto do PDF informado.');
  }

  const sourceName = path.basename(pdfPath);
  const baseDoc = new Document({
    pageContent: parsedPdf.text,
    metadata: { source: sourceName },
  });

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1200,
    chunkOverlap: 200,
  });

  const chunks = await splitter.splitDocuments([baseDoc]);

  if (chunks.length === 0) {
    throw new Error('Não foi possível gerar chunks do PDF.');
  }

  const embeddings = new OpenAIEmbeddings({
    model: 'text-embedding-3-small',
  });

  return MemoryVectorStore.fromDocuments(chunks, embeddings);
}

async function startInteractiveChat(vectorStore) {
  const llm = new ChatOpenAI({
    model: 'gpt-4.1-mini',
    temperature: 0.2,
  });

  const terminal = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log('\n✅ Base vetorial pronta. Faça perguntas sobre o PDF.');
  console.log("Digite 'sair' para encerrar.\n");

  try {
    while (true) {
      const question = (await terminal.question('Pergunta: ')).trim();

      if (!question) {
        continue;
      }

      if (['sair', 'exit', 'quit'].includes(question.toLowerCase())) {
        break;
      }

      const matches = await vectorStore.similaritySearch(question, 4);

      if (!matches.length) {
        console.log('\nResposta: Não encontrei contexto suficiente no PDF para responder.\n');
        continue;
      }

      const context = matches
        .map((doc, index) => `Trecho ${index + 1}:\n${doc.pageContent}`)
        .join('\n\n');

      const prompt = [
        'Você é um assistente de RAG focado em responder somente com base no contexto abaixo.',
        'Se a resposta não estiver no contexto, diga claramente que não encontrou a informação no documento.',
        '',
        `Contexto:\n${context}`,
        '',
        `Pergunta: ${question}`,
        'Resposta em português:',
      ].join('\n');

      const response = await llm.invoke(prompt);
      const answer = normalizeAiResponse(response);

      console.log(`\nResposta: ${answer}\n`);
    }
  } finally {
    terminal.close();
  }
}

async function main() {
  if (!process.env.OPENAI_API_KEY) {
    throw new Error('Defina OPENAI_API_KEY no ambiente (.env) para executar o assistente.');
  }

  const pdfPath = getCliPdfPath();
  console.log(`📄 Carregando PDF: ${pdfPath}`);

  const vectorStore = await createVectorStoreFromPdf(pdfPath);
  await startInteractiveChat(vectorStore);

  console.log('👋 Encerrado.');
}

main().catch((error) => {
  console.error(`❌ ${error.message}`);
  process.exitCode = 1;
});
