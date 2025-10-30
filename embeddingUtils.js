import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { LRUCache } from "lru-cache";

import dotenv from "dotenv";
dotenv.config();

let embeddingModel = null;

const embeddingCache = new LRUCache({
    max: 100,
    maxSize: 50_000_000,
    sizeCalculation: (value, key) => (value.length * 4) + key.length,
    ttl: 1000 * 60 * 60, // 1 hour
});

export async function initializeEmbeddingUtils({
    model = "gemini-embedding-001",
    apiKey = process.env.GOOGLE_GEMINI_API_KEY,
} = {}) {
    if (!apiKey) {
        throw new Error("Google API key is missing. Set it via environment variable or pass explicitly.");
    }

    embeddingModel = new GoogleGenerativeAIEmbeddings({
        model,
        apiKey,
    });

    embeddingCache.clear();

    return {
        modelName: model,
        provider: "Google Generative AI",
    };
}

// -------------------------------------
// -- Function to generate embeddings --
// -------------------------------------
export async function createEmbedding(text) {

    embeddingModel = new GoogleGenerativeAIEmbeddings({
        model: "gemini-embedding-001",
        apiKey: process.env.GOOGLE_GEMINI_API_KEY, // or better: use env variable
    });


    const cached = embeddingCache.get(text);
    if (cached) return cached;

    try {
        const vector = await embeddingModel.embedQuery(text);
        embeddingCache.set(text, vector);
        return vector;
    } catch (error) {
        console.error("Error generating Gemini embedding:", error);
        return [];
    }
}


export const tokenizer = (text) => {
    const tokens = text.split(/\s+/);
    return {
        input_ids: { size: tokens.length },
    };
};