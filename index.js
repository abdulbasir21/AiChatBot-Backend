const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");
dotenv.config();

const app = express();

app.use(cors({
  origin: "https://abdulbasir-portfolio.vercel.app",
  methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"],
  credentials: true
}));

app.use(express.json());

const { MongoClient } = require("mongodb");
const { MongoDBAtlasVectorSearch } = require("@langchain/mongodb");
const { HuggingFaceInferenceEmbeddings } = require("@langchain/community/embeddings/hf");
const { ChatGroq } = require("@langchain/groq");

const client = new MongoClient(process.env.MONGO_URI);
let collection;

async function getCollection() {
  if (collection) return collection;
  await client.connect();
  const db = client.db();
  collection = db.collection("documents");
  return collection;
}

const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: process.env.HUGGINGFACE_API_KEY,
  model: "sentence-transformers/all-MiniLM-L6-v2",
});

const llm = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "openai/gpt-oss-20b",
});

app.post("/chat", async (req, res) => {
  try {
    const { message } = req.body;

    if (!message) {
      return res.status(400).json({ error: "Message is required" });
    }

    const col = await getCollection();

    const vectorStore = new MongoDBAtlasVectorSearch(embeddings, {
      collection: col,
      indexName: "vector_index",
      textKey: "text",
      embeddingKey: "embedding",
    });

    const results = await vectorStore.similaritySearch(message, 4);
    const context = results.map((doc) => doc.pageContent).join("\n\n");

    const prompt = `
You are a helpful AI assistant.
Use ONLY the context below to answer the user's question.
If the answer is not found in the context, say:
"I could not find that information in the provided documents."

Context:
${context}

User Question:
${message}
`;

    const response = await llm.invoke(prompt);
    res.json({ reply: response.content });

  } catch (error) {
    console.log("❌ Error:", error);
    res.status(500).json({ error: "Something went wrong" });
  }
});

module.exports = app;  // ← Vercel needs this, NOT app.listen()
