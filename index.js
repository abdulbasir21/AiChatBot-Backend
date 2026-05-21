const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");
dotenv.config();

const app = express();

// ← CORS must be first, before everything
const corsOptions = {
  origin: "https://abdulbasir-portfolio.vercel.app",
  methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
  allowedHeaders: ["Content-Type", "Authorization"],
  credentials: true
};

app.use(cors(corsOptions));
app.options("/(.*)", cors(corsOptions)); // ← pass same options here too
app.use(express.json());

const { MongoClient } = require("mongodb");
const { MongoDBAtlasVectorSearch } = require("@langchain/mongodb");
const { HuggingFaceInferenceEmbeddings } = require("@langchain/community/embeddings/hf");
const { ChatGroq } = require("@langchain/groq");

const client = new MongoClient(process.env.MONGO_URI);
let collection;

// ← No connectDB() at bottom, connect inside route instead
async function getCollection() {
  if (collection) return collection; // ← reuse if already connected
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
  model: "llama-3.3-70b-versatile",
});

app.post("/chat", async (req, res) => {
  try {
    const { message } = req.body;

    if (!message) {
      return res.status(400).json({ error: "Message is required" });
    }

    const col = await getCollection(); // ← connect here instead

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

// ← For Vercel, export app instead of app.listen()
module.exports = app;
