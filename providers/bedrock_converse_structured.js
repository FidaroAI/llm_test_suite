const { BedrockRuntimeClient, ConverseCommand } = require('@aws-sdk/client-bedrock-runtime');

// promptfoo 0.121.x's built-in Bedrock Converse provider does not forward
// outputConfig, so this provider keeps the same request shape and adds it.
function parsePrompt(prompt) {
  try {
    const parsed = JSON.parse(prompt);
    if (Array.isArray(parsed)) {
      const messages = [];
      const system = [];

      for (const message of parsed) {
        const content = typeof message.content === 'string'
          ? [{ text: message.content }]
          : message.content;

        if (message.role === 'system') {
          system.push(...content.map((block) => (
            typeof block === 'string' ? { text: block } : { text: block.text ?? JSON.stringify(block) }
          )));
        } else if (message.role === 'user' || message.role === 'assistant') {
          messages.push({
            role: message.role,
            content: content.map((block) => (
              typeof block === 'string' ? { text: block } : { text: block.text ?? JSON.stringify(block) }
            )),
          });
        }
      }

      return {
        messages: messages.length ? messages : [{ role: 'user', content: [{ text: prompt }] }],
        ...(system.length ? { system } : {}),
      };
    }
  } catch {
    // Fall through to plain-text prompt handling.
  }

  return {
    messages: [{ role: 'user', content: [{ text: prompt }] }],
  };
}

function normalizeOutputConfig(outputConfig) {
  if (!outputConfig?.textFormat?.structure?.jsonSchema) {
    return outputConfig;
  }

  const normalized = structuredClone(outputConfig);
  const jsonSchema = normalized.textFormat.structure.jsonSchema;
  if (jsonSchema.schema && typeof jsonSchema.schema !== 'string') {
    jsonSchema.schema = JSON.stringify(jsonSchema.schema);
  }
  return normalized;
}

function extractOutput(response) {
  const blocks = response.output?.message?.content ?? [];
  return blocks
    .map((block) => {
      if (block.text) {
        return block.text;
      }
      if (block.toolUse) {
        return JSON.stringify({
          type: 'tool_use',
          id: block.toolUse.toolUseId,
          name: block.toolUse.name,
          input: block.toolUse.input,
        });
      }
      return '';
    })
    .filter(Boolean)
    .join('\n\n');
}

function extractReasoningBlocks(response) {
  const blocks = response.output?.message?.content ?? [];
  return blocks
    .map((block) => {
      const reasoning = block.reasoningContent;
      if (!reasoning) {
        return '';
      }
      if (reasoning.reasoningText?.text) {
        return reasoning.reasoningText.text;
      }
      if (reasoning.redactedContent) {
        return '[Redacted reasoning content]';
      }
      return '';
    })
    .filter(Boolean);
}

function extractStructuredContent(response) {
  const blocks = response.output?.message?.content ?? [];
  return blocks
    .map((block) => {
      if (block.text) {
        return { type: 'text', text: block.text };
      }
      const reasoning = block.reasoningContent;
      if (reasoning?.reasoningText?.text) {
        return { type: 'thinking', thinking: reasoning.reasoningText.text };
      }
      if (reasoning?.redactedContent) {
        return { type: 'thinking', thinking: '[Redacted reasoning content]' };
      }
      if (block.toolUse) {
        return {
          type: 'tool_use',
          id: block.toolUse.toolUseId,
          name: block.toolUse.name,
          input: block.toolUse.input,
        };
      }
      return null;
    })
    .filter(Boolean);
}

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function mergeConfig(baseConfig, overrideConfig) {
  const merged = { ...baseConfig };
  if (!isPlainObject(overrideConfig)) {
    return merged;
  }

  for (const [key, value] of Object.entries(overrideConfig)) {
    if (value === null) {
      delete merged[key];
    } else if (isPlainObject(value) && isPlainObject(merged[key])) {
      merged[key] = mergeConfig(merged[key], value);
    } else {
      merged[key] = value;
    }
  }
  return merged;
}

function buildConverseInput(config, modelId, prompt) {
  const { messages, system } = parsePrompt(prompt);
  const inferenceConfig = {
    ...(config.maxTokens ?? config.max_tokens
      ? { maxTokens: config.maxTokens ?? config.max_tokens }
      : {}),
    ...(config.temperature !== undefined ? { temperature: config.temperature } : {}),
    ...(config.topP ?? config.top_p
      ? { topP: config.topP ?? config.top_p }
      : {}),
    ...(config.stopSequences || config.stop
      ? { stopSequences: config.stopSequences || config.stop }
      : {}),
  };

  return {
    modelId,
    messages,
    ...(system ? { system } : {}),
    ...(Object.keys(inferenceConfig).length ? { inferenceConfig } : {}),
    ...(config.toolConfig ? { toolConfig: config.toolConfig } : {}),
    ...(config.outputConfig
      ? { outputConfig: normalizeOutputConfig(config.outputConfig) }
      : {}),
    ...(config.additionalModelRequestFields
      ? { additionalModelRequestFields: config.additionalModelRequestFields }
      : {}),
    ...(config.additionalModelResponseFieldPaths
      ? { additionalModelResponseFieldPaths: config.additionalModelResponseFieldPaths }
      : {}),
    ...(config.performanceConfig ? { performanceConfig: config.performanceConfig } : {}),
    ...(config.requestMetadata ? { requestMetadata: config.requestMetadata } : {}),
  };
}

class BedrockConverseStructuredProvider {
  constructor(options = {}) {
    this.config = options.config || {};
    this.modelId = this.config.modelId || process.env.BEDROCK_MODEL_ID;
    this.providerId = `bedrock:converse-structured:${this.modelId || 'unknown-model'}`;
  }

  id() {
    return this.providerId;
  }

  async callApi(prompt, context) {
    const perTestConfig = context?.test?.options?.bedrock || {};
    const effectiveConfig = mergeConfig(this.config, perTestConfig);
    const modelId = effectiveConfig.modelId || this.modelId;

    if (!modelId) {
      return { error: 'BEDROCK_MODEL_ID or config.modelId is required' };
    }

    const input = buildConverseInput(effectiveConfig, modelId, prompt);

    try {
      const client = new BedrockRuntimeClient({ region: effectiveConfig.region || process.env.AWS_REGION });
      const response = await client.send(new ConverseCommand(input));
      const reasoningBlocks = extractReasoningBlocks(response);
      const reasoningContent = reasoningBlocks.join('\n\n');
      return {
        output: extractOutput(response),
        content: extractStructuredContent(response),
        ...(reasoningContent ? { reasoning_content: reasoningContent, thinking: reasoningContent } : {}),
        tokenUsage: response.usage
          ? {
            prompt: response.usage.inputTokens,
            completion: response.usage.outputTokens,
            total: response.usage.totalTokens,
          }
          : undefined,
        metadata: {
          stopReason: response.stopReason,
          metrics: response.metrics,
        },
      };
    } catch (error) {
      return { error: `Bedrock Converse API error: ${error.message || String(error)}` };
    }
  }
}

BedrockConverseStructuredProvider.buildConverseInput = buildConverseInput;

module.exports = BedrockConverseStructuredProvider;
