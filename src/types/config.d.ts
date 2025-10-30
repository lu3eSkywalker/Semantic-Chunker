declare module './config.js' {
    export const DEFAULT_CONFIG: {
        LOGGING: boolean,
        MAX_TOKEN_SIZE: number,
        SIMILARITY_THRESHOLD: number,
        DYNAMIC_THRESHOLD_LOWER_BOUND: number,
        DYNAMIC_THRESHOLD_UPPER_BOUND: number,
        NUM_SIMILARITY_SENTENCES_LOOKAHEAD: number,
        COMBINE_CHUNKS: boolean,
        COMBINE_CHUNKS_SIMILARITY_THRESHOLD: number,
        RETURN_EMBEDDING: boolean,
        RETURN_TOKEN_LENGTH: boolean,
        CHUNK_PREFIX: null,
        EXCLUDE_CHUNK_PREFIX_IN_RESULTS: boolean,
    }
}