# 🚀 RAG PDF Assistant com Node.js + LangChain

Este projeto demonstra na prática como construir um sistema de IA baseado em **RAG (Retrieval-Augmented Generation)** utilizando:

- Node.js
- LangChain
- OpenAI
- Embeddings Vetoriais
- PDF Parsing
- Vector Store em memória

O sistema permite carregar um PDF, indexar seu conteúdo semanticamente e realizar perguntas diretamente no terminal utilizando IA contextualizada.

## 🔥 Funcionalidades

- 📄 Leitura automática de PDFs
- ✂️ Chunking inteligente de texto
- 🧠 Embeddings com OpenAI
- 🔎 Busca semântica contextual
- 💬 Chat interativo no terminal
- ⚡ Respostas utilizando GPT-4.1-mini

## 🛠️ Tecnologias

- Node.js
- LangChain
- OpenAI API
- pdf-parse
- MemoryVectorStore

## 📌 Objetivo

Demonstrar de forma simples e prática como criar aplicações modernas de IA com recuperação de contexto usando documentos reais.

## ✅ Pré-requisitos

- Node.js 20+
- Chave da OpenAI (`OPENAI_API_KEY`)

## ▶️ Como executar

1. Instale as dependências:

   ```bash
   npm install
   ```

2. Configure as variáveis de ambiente:

   ```bash
   cp .env.example .env
   ```

   Edite o arquivo `.env` e informe sua chave da OpenAI.

3. Execute o assistente passando o caminho do PDF:

   ```bash
   npm start -- ./caminho/arquivo.pdf
   ```

4. Faça perguntas no terminal. Para sair, digite `sair`.

## 🧪 Testes

```bash
npm test
```
