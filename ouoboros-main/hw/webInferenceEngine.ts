/**
 * Web Inference Engine - Adapted for Ouroboros Architecture
 * Integrates with Hugging Face Inference API for web-based model inference
 */

export interface InferenceConfig {
    model: string;
    apiKey?: string;
    temperature?: number;
    maxTokens?: number;
    topP?: number;
}

export interface InferenceResult {
    text: string;
    metrics: {
        latency: number;
        tokensGenerated: number;
        tokensPerSecond: number;
    };
}

export class WebInferenceEngine {
    private apiKey: string;
    private baseUrl: string = 'https://api-inference.huggingface.co/models';
    private currentModel: string = '';

    constructor(apiKey: string = '') {
        this.apiKey = apiKey;
    }

    setApiKey(apiKey: string): void {
        this.apiKey = apiKey;
    }

    /**
     * Search for models on Hugging Face
     */
    async searchModels(query: string, limit: number = 10): Promise<any[]> {
        try {
            const response = await fetch(
                `https://huggingface.co/api/models?search=${encodeURIComponent(query)}&limit=${limit}`
            );
            
            if (!response.ok) {
                throw new Error(`Search failed: ${response.statusText}`);
            }

            const models = await response.json();
            return models.map((model: any) => ({
                id: model.modelId,
                name: model.modelId,
                likes: model.likes || 0,
                downloads: model.downloads || 0,
                tags: model.tags || [],
                pipeline_tag: model.pipeline_tag || 'text-generation'
            }));
        } catch (error) {
            console.error('Model search error:', error);
            throw error;
        }
    }

    /**
     * Get model details
     */
    async getModelDetails(modelId: string): Promise<any> {
        try {
            const response = await fetch(`https://huggingface.co/api/models/${modelId}`);
            
            if (!response.ok) {
                throw new Error(`Failed to get model details: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            console.error('Model details error:', error);
            throw error;
        }
    }

    /**
     * Run inference using Hugging Face Inference API
     */
    async inference(prompt: string, config: InferenceConfig): Promise<InferenceResult> {
        const startTime = performance.now();
        
        try {
            const headers: HeadersInit = {
                'Content-Type': 'application/json'
            };

            if (this.apiKey) {
                headers['Authorization'] = `Bearer ${this.apiKey}`;
            }

            const body = {
                inputs: prompt,
                parameters: {
                    temperature: config.temperature || 0.7,
                    max_new_tokens: config.maxTokens || 100,
                    top_p: config.topP || 0.95,
                    return_full_text: false
                }
            };

            const response = await fetch(
                `${this.baseUrl}/${config.model}`,
                {
                    method: 'POST',
                    headers,
                    body: JSON.stringify(body)
                }
            );

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Inference failed: ${response.statusText} - ${errorText}`);
            }

            const result = await response.json();
            const endTime = performance.now();
            const latency = endTime - startTime;

            // Handle different response formats
            let generatedText = '';
            if (Array.isArray(result)) {
                generatedText = result[0]?.generated_text || result[0]?.text || '';
            } else if (typeof result === 'object') {
                generatedText = result.generated_text || result.text || JSON.stringify(result);
            } else {
                generatedText = String(result);
            }

            // Remove the prompt from the response if present
            if (generatedText.startsWith(prompt)) {
                generatedText = generatedText.substring(prompt.length).trim();
            }

            const tokensGenerated = this.estimateTokenCount(generatedText);
            const tokensPerSecond = latency > 0 ? (tokensGenerated / (latency / 1000)) : 0;

            return {
                text: generatedText,
                metrics: {
                    latency,
                    tokensGenerated,
                    tokensPerSecond
                }
            };
        } catch (error) {
            console.error('Inference error:', error);
            throw error;
        }
    }

    /**
     * Stream inference (for real-time token generation)
     */
    async *streamInference(prompt: string, config: InferenceConfig): AsyncGenerator<string, void, unknown> {
        try {
            const headers: HeadersInit = {
                'Content-Type': 'application/json'
            };

            if (this.apiKey) {
                headers['Authorization'] = `Bearer ${this.apiKey}`;
            }

            const body = {
                inputs: prompt,
                parameters: {
                    temperature: config.temperature || 0.7,
                    max_new_tokens: config.maxTokens || 100,
                    top_p: config.topP || 0.95,
                    stream: true
                }
            };

            const response = await fetch(
                `${this.baseUrl}/${config.model}`,
                {
                    method: 'POST',
                    headers,
                    body: JSON.stringify(body)
                }
            );

            if (!response.ok) {
                throw new Error(`Stream inference failed: ${response.statusText}`);
            }

            const reader = response.body?.getReader();
            if (!reader) {
                throw new Error('Response body is not readable');
            }

            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        if (data === '[DONE]') return;

                        try {
                            const parsed = JSON.parse(data);
                            const token = parsed.token?.text || parsed.generated_text || '';
                            if (token) yield token;
                        } catch (e) {
                            // Skip invalid JSON
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Stream inference error:', error);
            throw error;
        }
    }

    /**
     * Estimate token count (rough approximation)
     */
    private estimateTokenCount(text: string): number {
        // Rough estimate: ~4 characters per token
        return Math.ceil(text.length / 4);
    }

    /**
     * Check if a model supports text generation
     */
    async checkModelCapability(modelId: string): Promise<boolean> {
        try {
            const details = await this.getModelDetails(modelId);
            const pipelineTag = details.pipeline_tag || details.tags?.find((t: string) => t.startsWith('text-'));
            return pipelineTag === 'text-generation' || 
                   pipelineTag === 'text2text-generation' ||
                   details.tags?.includes('text-generation');
        } catch (error) {
            return false;
        }
    }
}

export const webInferenceEngine = new WebInferenceEngine();
