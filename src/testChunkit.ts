// testChunkit.js
import { chunkit } from './chunkitGoogleGemini.js';


export interface DocumentInput {
    document_text: string;
    document_name?: string;
}

async function main() {
    const documents: any = [
        {
            document_name: "Test Document",
            document_text: `
To make pancakes, first mix flour, milk, and eggs in a bowl. 
Then heat a non-stick pan and pour a small amount of the batter. 
Cook until golden brown on both sides. Serve with maple syrup.
Electric vehicles (EVs) are automobiles that use electric motors instead of internal combustion engines. 
They rely on rechargeable batteries for power. 
While EVs help reduce greenhouse gas emissions, they still face challenges like limited range and long charging times. 
Governments worldwide are offering incentives to promote EV adoption.
Neural networks are inspired by the structure of the human brain. 
They consist of interconnected neurons organized into layers. 
Training involves adjusting weights through backpropagation to minimize error. 
However, large models require significant computational resources, which makes training expensive. 
Recent research explores more efficient architectures like transformers that reduce training time without compromising accuracy.
            `
        }
    ];

    try {
        const results = await chunkit(documents, {
            logging: true,
            returnEmbedding: true,
            returnTokenLength: true,
        });

        console.log("\n\n========== Final Chunked Output ==========");
        results.forEach((chunk: any) => {
            console.log("Chunk #", chunk.chunk_number);
            console.log("Text:", chunk.text);
            console.log("Token length:", chunk.token_length);
            console.log("Embedding vector length:", chunk.embedding ? chunk.embedding.length : "N/A");
            console.log("--------------------------------------\n");
        });
    } catch (err) {
        console.error("❌ Error in chunkit():", err);
        console.dir(err, { depth: null });
    }

}

main().catch(console.error);
