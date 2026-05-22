// Carrega variáveis de ambiente do arquivo `.env` para `process.env`.
import "dotenv/config";

// Funções de sistema de arquivos (ler/escrever arquivos locais).
import fs from "fs";

// Interface readline baseada em promises para entrada/saída no terminal.
import readline from "node:readline/promises";

// Biblioteca para extrair texto de arquivos PDF.
import pdf from "pdf-parse/lib/pdf-parse.js";

// Clientes LangChain/OpenAI: modelo de chat e gerador de embeddings.
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";

// Splitter que divide texto em pedaços (chunks) por caracteres.
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

// Vector store em memória para armazenar embeddings (bom para dev/testes).
import { MemoryVectorStore } from "@langchain/classic/vectorstores/memory";

/*
  Script RAG (Retrieval-Augmented Generation) básico:
  - Extrai texto de um PDF, divide em chunks, gera embeddings,
    armazena em um Vector Store em memória, e permite perguntas
    interativas no terminal usando o contexto retornado pelo retriever.
*/

/* Função de Leitura do PDF
   - Lê o arquivo binário do caminho fornecido (`pdfPath`).
   - Usa a biblioteca `pdf-parse` para extrair o texto bruto do PDF.
   - Retorna uma string com todo o texto (ou string vazia se não houver).
*/
async function loadPdfText(pdfPath) {
  const pdfBuffer = fs.readFileSync(pdfPath);
  const parsed = await pdf(pdfBuffer);
  return parsed.text || "";
}

/* Configuração de chunking
  - Quebra o texto em pedaços menores (chunks) para que os embeddings
    capturem contexto local e para controlar o tamanho do prompt passado ao modelo.
*/
const PDF_PATH = "file.pdf";
const CHUNK_SIZE = 1200;
const CHUNK_OVERLAP = 200;

/* Split do Texto
   - Carrega o texto completo do PDF e cria um `RecursiveCharacterTextSplitter`.
   - O splitter divide o texto respeitando `chunkSize` e `chunkOverlap`.
   - Resultado: array `chunks` com strings menores representando partes do PDF.
*/

const pdfText = await loadPdfText(PDF_PATH);

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP,
});

const chunks = await splitter.splitText(pdfText);

/* Embeddings
   - Transforma cada chunk de texto em um vetor (embedding) usando o OpenAI
     (`text-embedding-3-small`).
   - Esses vetores permitem calcular similaridade entre pergunta e trechos do PDF.
*/
const embeddings = new OpenAIEmbeddings({
  model: "text-embedding-3-small",
  openAIApiKey: process.env.OPENAI_API_KEY,
});

/* Banco Vetorial (Vector Store)
   - Guarda embeddings na memória (`MemoryVectorStore`) com referência ao texto.
   - `fromTexts` recebe os textos (chunks) e metadados (aqui, objetos vazios).
   - Em um sistema em produção, troque por um Vector DB persistente (pinecone, weaviate, etc.).
*/
const vectorStore = await MemoryVectorStore.fromTexts(
  chunks,
  chunks.map(() => ({})),
  embeddings
);

/* Retriever
   - O retriever encapsula a lógica de busca por similaridade no Vector Store.
   - Aqui definimos `TOP_K` para quantos trechos relevantes queremos retornar.
*/
const TOP_K = 4;
const retriever = vectorStore.asRetriever(TOP_K);

/* Modelo de Chat (GPT)
   - Instancia um cliente `ChatOpenAI` para gerar respostas com contexto.
   - `temperature: 0` busca respostas mais determinísticas.
*/
const model = new ChatOpenAI({
  model: "gpt-4.1-mini",
  temperature: 0,
  openAIApiKey: process.env.OPENAI_API_KEY,
});


/* Interface de Chat no Terminal
   - Usamos `readline` para criar uma REPL simples que lê perguntas do usuário.
   - O loop principal envia cada pergunta ao retriever + modelo e imprime a resposta.
*/
console.log("PDF carregado!");

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

/* answerQuestion(question, retriever, model)
   - Fluxo principal para responder uma pergunta:
     1) Usa o retriever para obter documentos/chunks relevantes para a `question`.
     2) Concatena o conteúdo desses documentos em `context`.
     3) Monta um `prompt` que instrui o modelo a responder usando o contexto.
     4) Chama o modelo de chat e exibe a `response` no terminal.
   - Observação: em produção, seria ideal adicionar instruções de sistema,
     limites de tokens e tratamento de erros/respostas vazias.
*/
async function answerQuestion(question, retriever, model) {

  const relevantDocs = await retriever.invoke(question);

  const context = relevantDocs
    .map(doc => doc.pageContent)
    .join("\n\n");

  const prompt = `
Você é um assistente que responde perguntas usando o contexto abaixo.

Contexto:
${context}

Pergunta:
${question}
`;
const response = await model.invoke(prompt);

  console.log("\nResposta:\n");

  console.log(response.content);
}

/* Loop Principal
   - Mantém a REPL rodando até que o usuário escreva 'sair', 'exit' ou 'quit'.
   - Normaliza a pergunta, evita entradas vazias e chama `answerQuestion`.
   - No `finally` garantimos que `rl.close()` seja chamado ao encerrar.
*/
try {

  while (true) {

    const question = (
      await rl.question("Pergunta> ")
    ).trim();

    if (!question) continue;

    const normalized = question.toLowerCase();

    if (
      normalized === "sair" ||
      normalized === "exit" ||
      normalized === "quit"
    ) {
      break; // Encerra o loop quando o usuário pede para sair
    }

    await answerQuestion(
      question,
      retriever,
      model
    );

    console.log("\n----------------\n");
  }

} finally {

  rl.close();
}