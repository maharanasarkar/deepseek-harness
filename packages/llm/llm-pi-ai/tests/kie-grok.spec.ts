import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { createUserMessage, ReasoningEffortId } from '@deepseek-ai/dsh-llm'
import * as LlmPiAi from '@deepseek-ai/dsh-llm-pi-ai'
import { resolveProfiles } from '../src/config.ts'
import { assemble } from './assemble.ts'
import { closeMockServers, mockServer } from './mock-server.ts'

afterEach(async () => {
  vi.unstubAllEnvs()
  await closeMockServers()
})

beforeEach(() => {
  vi.stubEnv('KIE_API_KEY', 'test-key')
})

/** Kie Grok route shape: nested per-family baseURL under openai-responses.
 *
 * Kie has no true "off" (minimum effort is low, omitted defaults to low),
 * while pi-ai sends effort "none" for an unspelled off, which Kie rejects.
 * The profile therefore defaults every effort-less request to low and spells
 * off as low.
 */
function kieProviders(baseURL: string): Record<string, LlmPiAi.PiAiProviderProfile> {
  return {
    'kie-grok': {
      displayName: 'Kie Grok',
      apiKeyEnv: 'KIE_API_KEY',
      api: 'openai-responses',
      baseURL,
      reasoning: 'low',
      models: [{
        id: 'grok-4-7',
        name: 'Grok 4.7 (Kie)',
        contextWindow: 500000,
        reasoningEfforts: { off: 'low', low: 'low', medium: 'medium', high: 'high', xhigh: 'xhigh' },
      }],
    },
  }
}

/** Kie wire vocabulary: named SSE events ending in response.completed. */
const kieFrames = [
  'event: response.output_item.added',
  'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_1","role":"assistant","content":[]}}',
  '',
  'event: response.output_text.delta',
  'data: {"type":"response.output_text.delta","output_index":0,"delta":"Hello"}',
  '',
  'event: response.completed',
  'data: {"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}',
  '',
  'data: [DONE]',
  '',
]

describe('kie-grok gateway route', () => {
  it('resolves the overlay profile without a catalog entry', () => {
    const profiles = resolveProfiles(kieProviders('https://api.kie.ai/grok/v1'))
    expect(profiles.get('kie-grok')?.displayName).toBe('Kie Grok')
  })

  it('streams grok-4-7 through the family baseURL with the selected effort', async () => {
    const server = await mockServer([{ rawFrames: kieFrames }])
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmPiAi, { providers: kieProviders(`${server.url}/grok/v1`) })
    const result = await assemble(ctx, {
      provider: 'kie-grok',
      model: 'grok-4-7',
      reasoningEffort: ReasoningEffortId('xhigh'),
      messages: [createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } })],
    })
    expect(server.paths).toEqual(['/grok/v1/responses'])
    expect(server.headers[0]?.authorization).toBe('Bearer test-key')
    const request = server.requests[0] as Record<string, unknown>
    expect(request['model']).toBe('grok-4-7')
    expect(request['reasoning']).toEqual({ effort: 'xhigh', summary: 'auto' })
    expect(result.message.content).toEqual([{ type: 'text', text: 'Hello' }])
    expect(result.usage).toEqual({ inputTokens: 10, outputTokens: 5, totalTokens: 15 })
    expect(result.finish).toEqual({ kind: 'stop' })
  })

  it('defaults effort-less requests to low, never pi-ai’s "none"', async () => {
    const server = await mockServer([{ rawFrames: kieFrames }, { rawFrames: kieFrames }])
    const ctx = new Context()
    await ctx.plugin(LlmRuntime)
    await ctx.plugin(LlmPiAi, { providers: kieProviders(`${server.url}/grok/v1`) })
    for (const effort of [undefined, ReasoningEffortId('off')] as const) {
      const result = await assemble(ctx, {
        provider: 'kie-grok',
        model: 'grok-4-7',
        ...effort === undefined ? {} : { reasoningEffort: effort },
        messages: [createUserMessage({ content: [{ type: 'text', text: 'hi' }], source: { kind: 'user' } })],
      })
      expect(result.finish).toEqual({ kind: 'stop' })
    }
    const bodies = server.requests as Record<string, unknown>[]
    expect(bodies).toHaveLength(2)
    for (const body of bodies) {
      // The profile default and the explicit off both land on Kie-legal low;
      // pi-ai's unspelled-off "none" must never reach the wire.
      expect((body['reasoning'] as { effort?: string })?.effort).toBe('low')
    }
  })
})
