import { parseSentences } from 'sentence-parse';
import { DEFAULT_CONFIG } from './config.js';
import { computeAdvancedSimilarities, adjustThreshold } from './similarityUtils.js';
import { createChunks, optimizeAndRebalanceChunks, applyPrefixToChunk } from './chunkingUtils.js';
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { tokenizer } from './embeddingUtils.js';

import dotenv from "dotenv";
dotenv.config();

interface ChunkResult {
    document_id: number,
    document_name: string,
    number_of_chunks: number,
    chunk_number: number,
    model_name: string,
    text: string,
    embedding?: number[],
    token_length?: number;
}

export interface DocumentInput {
    document_text: string;
    document_name?: string;
}

// ---------------------------
// -- Main chunkit function --
// ---------------------------
export async function chunkit(
    documents: DocumentInput[],
    {
        logging = DEFAULT_CONFIG.LOGGING,
        maxTokenSize = DEFAULT_CONFIG.MAX_TOKEN_SIZE,
        similarityThreshold = DEFAULT_CONFIG.SIMILARITY_THRESHOLD,
        dynamicThresholdLowerBound = DEFAULT_CONFIG.DYNAMIC_THRESHOLD_LOWER_BOUND,
        dynamicThresholdUpperBound = DEFAULT_CONFIG.DYNAMIC_THRESHOLD_UPPER_BOUND,
        numSimilaritySentencesLookahead = DEFAULT_CONFIG.NUM_SIMILARITY_SENTENCES_LOOKAHEAD,
        combineChunks = DEFAULT_CONFIG.COMBINE_CHUNKS,
        combineChunksSimilarityThreshold = DEFAULT_CONFIG.COMBINE_CHUNKS_SIMILARITY_THRESHOLD,
        returnEmbedding = DEFAULT_CONFIG.RETURN_EMBEDDING,
        returnTokenLength = DEFAULT_CONFIG.RETURN_TOKEN_LENGTH,
        chunkPrefix = DEFAULT_CONFIG.CHUNK_PREFIX,
        excludeChunkPrefixInResults = false,
    } = {}) {

    if (!Array.isArray(documents)) {
        throw new Error('Input must be an array of document objects');
    }

    // ------------------------------
    // Initialize Gemini embeddings
    // ------------------------------
    const embeddingModel = new GoogleGenerativeAIEmbeddings({
        model: "gemini-embedding-001",
        apiKey: process.env.GOOGLE_GEMINI_API_KEY,
    });

    const modelName = "gemini-embedding-001";

    const allResults = await Promise.all(documents.map(async (doc) => {
        if (!doc.document_text) throw new Error('Each document must have a document_text property');

        let normalizedText = doc.document_text.replace(/([^\n])\n([^\n])/g, '$1 $2');
        normalizedText = normalizedText.replace(/\s{2,}/g, ' ');
        doc.document_text = normalizedText;

        const sentences = await parseSentences(doc.document_text);

        const { similarities, average, variance } = await computeAdvancedSimilarities(
            sentences,
            { numSimilaritySentencesLookahead, logging }
        );

        let dynamicThreshold = similarityThreshold;
        if (average != null && variance != null) {
            dynamicThreshold = adjustThreshold(
                average, variance, similarityThreshold,
                dynamicThresholdLowerBound, dynamicThresholdUpperBound
            );
        }

        const initialChunks = createChunks(sentences, similarities, maxTokenSize, dynamicThreshold, logging);

        if (logging) {
            console.log('\n=============\ninitialChunks\n=============');
            initialChunks.forEach((chunk, i) => {
                console.log(`\n-- Chunk ${i + 1} --`);
                console.log(chunk.substring(0, 50) + '...');
            });
        }

        let finalChunks = combineChunks
            ? await optimizeAndRebalanceChunks(initialChunks, tokenizer, maxTokenSize, combineChunksSimilarityThreshold)
            : initialChunks;

        const documentId = Date.now();
        const documentName = doc.document_name || "";
        const numberOfChunks = finalChunks.length;

        return Promise.all(finalChunks.map(async (chunk, index) => {
            const prefixedChunk = applyPrefixToChunk(chunkPrefix, chunk);
            const result: ChunkResult = {
                document_id: documentId,
                document_name: documentName,
                number_of_chunks: numberOfChunks,
                chunk_number: index + 1,
                model_name: modelName,
                text: prefixedChunk,
            };

            // ---- Gemini Embeddings ----
            if (returnEmbedding) {
                try {
                    const vector = await embeddingModel.embedQuery(prefixedChunk);
                    result.embedding = vector;
                } catch (error) {
                    console.error("Embedding generation failed:", error);
                    result.embedding = [];
                }
            }

            // ---- Token length (approximation) ----
            if (returnTokenLength) {
                result.token_length = prefixedChunk.split(/\s+/).length;
            }

            if (excludeChunkPrefixInResults && chunkPrefix && chunkPrefix.trim()) {
                const prefixPattern = new RegExp(`^${chunkPrefix}:\\s*`);
                result.text = result.text.replace(prefixPattern, '');
            }

            return result;
        }));
    }));

    return allResults.flat();
}