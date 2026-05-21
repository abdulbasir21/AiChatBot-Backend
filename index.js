const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");

dotenv.config();

const app = express();

app.use(cors({
  origin: "https://abdulbasir-portfolio.vercel.app/",
  methods: ["POST"],
  credentials: true
}));
app.use(express.json());

/* -----------------------------
   MongoDB Native Driver
------------------------------*/
const { MongoClient } = require("mongodb");

/* -----------------------------
   LangChain Imports
------------------------------*/
const { MongoDBAtlasVectorSearch } = require("@langchain/mongodb");

const {
  HuggingFaceInferenceEmbeddings,
} = require("@langchain/community/embeddings/hf");

const { ChatGroq } = require("@langchain/groq");

/* -----------------------------
   MongoDB Connection (Single shared client)
------------------------------*/
const client = new MongoClient(process.env.MONGO_URI);
let collection;

async function connectDB() {
  await client.connect();
  const db = client.db();
  collection = db.collection("documents");
  console.log("✅ MongoDB connected");
}

/* -----------------------------
   Embedding Model
------------------------------*/
const embeddings = new HuggingFaceInferenceEmbeddings({
  apiKey: process.env.HUGGINGFACE_API_KEY,
  model: "sentence-transformers/all-MiniLM-L6-v2",
});

/* -----------------------------
   Groq LLM
------------------------------*/
const llm = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: "llama-3.3-70b-versatile",
});

/* -----------------------------
   Chat Route (RAG)
------------------------------*/
app.post("/chat", async (req, res) => {

  try {

    const { message } = req.body;

    if (!message) {
      return res.status(400).json({
        error: "Message is required",
      });
    }

    console.log("🧠 User Question:", message);

    /* -----------------------------
       Connect Vector Store
    ------------------------------*/
    const vectorStore = new MongoDBAtlasVectorSearch(
      embeddings,
      {
        collection,
        indexName: "vector_index",
        textKey: "text",
        embeddingKey: "embedding",
      }
    );

    /* -----------------------------
       Similarity Search
    ------------------------------*/
    const results = await vectorStore.similaritySearch(
      message,
      4
    );

    console.log("📚 Retrieved Chunks:", results.length);

    /* -----------------------------
       Build Context
    ------------------------------*/
    const context = results
      .map((doc) => doc.pageContent)
      .join("\n\n");

    /* -----------------------------
       Final Prompt
    ------------------------------*/
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

    /* -----------------------------
       Ask Groq LLM
    ------------------------------*/
    const response = await llm.invoke(prompt);

    console.log("✅ Response generated");

    res.json({
      reply: response.content,
    });

  } catch (error) {

    console.log("❌ Error:", error);

    res.status(500).json({
      error: "Something went wrong",
    });

  }
});

/* -----------------------------
   Start Server (connect DB first)
------------------------------*/
connectDB().then(() => {
  app.listen(5000, () => {
    console.log("🚀 Server running on port 5000");
  });
}).catch((err) => {
  console.error("❌ Failed to connect to MongoDB:", err);
  process.exit(1);
});
