# 🚀 RAG PDF Assistant com Node.js + LangChain

Este projeto demonstra na prática como construir um sistema de IA baseado em RAG (Retrieval-Augmented Generation) utilizando:

- Node.js
- LangChain
- OpenAI
- Embeddings vetoriais
- Parsing de PDF
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

- Node.js 18+
- Chave da OpenAI em `OPENAI_API_KEY`

## ⚙️ Instalação

```bash
npm install
cp .env.example .env
```

Edite o `.env` e configure sua chave da OpenAI.

## ▶️ Uso

```bash
npm start -- ./caminho/para/documento.pdf
```

Após o carregamento do PDF, o terminal entra em modo interativo para perguntas.

Para sair do chat, digite:

```text
sair
```
