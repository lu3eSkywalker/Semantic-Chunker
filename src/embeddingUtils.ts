import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { LRUCache } from "lru-cache";

import dotenv from "dotenv";
dotenv.config();

let embeddingModel: GoogleGenerativeAIEmbeddings;

const embeddingCache = new LRUCache<string, number[]>({
    max: 100,
    maxSize: 50_000_000,
    sizeCalculation: (value, key) => (value.length * 4) + key.length,
    ttl: 1000 * 60 * 60,
});

// -------------------------------------
// -- Function to generate embeddings --
// -------------------------------------
export async function createEmbedding(text: string): Promise<number[]> {

    embeddingModel = new GoogleGenerativeAIEmbeddings({
        model: "gemini-embedding-001",
        apiKey: process.env.GOOGLE_GEMINI_API_KEY,
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

interface TokenizerOutput {
    input_ids: { size: number };
}

export const tokenizer = (text: string): TokenizerOutput => {
    const tokens = text.split(/\s+/);
    return {
        input_ids: { size: tokens.length },
    };
};