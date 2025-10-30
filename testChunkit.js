import { chunkit } from './chunkitGoogleGemini.js';

async function main() {
    const documents = [
        {
            document_name: "Test Document",
            document_text: `
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. 
Ethereum is a decentralized platform that enables developers to build 
and deploy smart contracts and decentralized applications (DApps) on a 
blockchain. Unlike Bitcoin, which is primarily a digital currency, 
Ethereum’s blockchain is more flexible,The sun was bright, and the air smelled of earth and fresh grass. 
The Indian Premier League(IPL) is the biggest cricket league in the world. 
People all over the world watch the matches and cheer for their favourite teams.

Terrorism is a big danger to peace and safety. 
It causes harm to people and creates fear in cities and villages. 
When such attacks happen, they leave behind pain and sadness. 
To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
            `
        }
    ];

    // Call chunkit
    const results = await chunkit(documents, {
        logging: true,
        returnEmbedding: true,
        returnTokenLength: true
    });

    console.log("\n\n========== Final Chunked Output ==========");
    results.forEach((chunk) => {
        console.log("Chunk #", chunk.chunk_number);
        console.log("Text:", chunk.text);
        console.log("Token length:", chunk.token_length);
        console.log("Embedding vector length:", chunk.embedding ? chunk.embedding.length : "N/A");
        console.log("--------------------------------------\n");
    });
}

main().catch(console.error);