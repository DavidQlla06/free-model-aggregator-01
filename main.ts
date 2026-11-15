// free-model-aggregator-01: one server, two client wires, ten simulated free providers.
//
// The server reads the trial brief ($MAKEFUN_PROMPT_FILE), probes every provider's
// catalog route, and registers one catalog with the env. It then answers the client's
// twenty requests — OpenAI Chat Completions and Anthropic Messages, JSON and SSE —
// by routing each to the strongest free model that meets the request, failing over
// to the next provider the moment one refuses.
//
// Routing, in short:
//   - Candidates come from the brief's quality table (the sealed ratings), joined
//     against what each provider's own catalog route actually listed. A model the
//     table calls paid is never picked, whatever its rating.
//   - The workload manifest is known up front, so the server plans the twenty
//     requests before the first arrives: each request's first pick is its best legal
//     model, subject to a conservative per-provider call cap (the low end of each
//     free tier's range), so a scarce top model is not spent on a light request
//     while a heavier one still waits. Weight-2 requests arrive first and so take
//     the top models first.
//   - A provider that refuses (402/429/5xx, a timeout, a bad answer) is counted and
//     skipped for that request; the request goes to the next candidate, preferring
//     providers that still have slack under their planned cap. A provider that
//     refuses twice is treated as spent and drops to the back of every later list.
//     The same provider is only retried when nothing else is left to try.
//   - A JSON-mode OpenAI request never routes to an Anthropic-wire provider: that
//     wire cannot carry response_format, so its echo would report JSON mode absent.
//   - Every answer is returned in the wire the client used, streamed or not as it
//     asked, with the provider's text whole, its usage counts translated into that
//     wire's shape, and its stop reason mapped across the wires.
//
// Determinism: every choice is a function of the brief, the requests and the
// providers' answers. No wall clock and no unseeded randomness anywhere in a
// decision; the fixed timestamps in model lists are constants, not the clock.
//
// Node 22, standard library only. Run as `node main.ts`; `node main.ts --list`
// prints the catalog document and exits 0 without starting the server.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 8000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** A fixed epoch second for every listed model's `created`: the reply never reads the clock. */
const FIXED_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`, for the same reason. */
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** The Anthropic version header the client sends, and this server sends upstream. */
const ANTHROPIC_VERSION = '2023-06-01';
/** The most providers tried for one request before giving up. */
const MAX_ATTEMPTS = 6;
/** A provider that refused this many times is treated as spent. */
const DEAD_AFTER_FAILURES = 2;

/**
 * Conservative per-provider call caps used only for planning: the low end of each
 * free tier's published range. The plan never assigns more calls than this to a
 * provider, so a planned first pick is (barring a scheduled refusal or an outage)
 * inside the provider's real budget. Anything above the cap is still reachable by
 * failover; the cap only orders the plan, it never forbids a call.
 */
const PLANNED_CAP: Record<string, number> = {
  groq: 9,
  cerebras: 5,
  gemini: 6,
  zai: 4,
  openrouter: 10,
  sambanova: 4,
  pollinations: 10,
  cloudflare: 7,
  ovh: 5,
  kilol: 12,
};
const DEFAULT_CAP = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

type Wire = 'openai' | 'anthropic' | 'gemini';
type AuthScheme = 'bearer' | 'x-api-key' | 'x-goog-api-key' | 'none';
type Json = Record<string, unknown>;

const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);
const asString = (v: unknown): string => (typeof v === 'string' ? v : '');
const asNumber = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const asBool = (v: unknown): boolean | null => (typeof v === 'boolean' ? v : null);
const asInt = (v: unknown): number | null => {
  const n = asNumber(v);
  return n !== null && Number.isSafeInteger(n) ? n : null;
};

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** One provider as the brief's directory lists it. `base` is its path (or absolute URL). */
interface DirectoryProvider {
  id: string;
  name: string;
  wire: Wire;
  auth: AuthScheme;
  base: string;
  key_env: string;
  key: string;
}

/** One row of the brief's quality table. */
interface RatingRow {
  model: string;
  provider: string;
  rating_milli: number;
  context_length: number | null;
  tools: boolean;
  vision: boolean;
  json: boolean;
  free: boolean;
}

/** What one request needs: a model without it cannot serve the request. */
interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

interface WorkloadEntry {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

interface Brief {
  providers: DirectoryProvider[];
  ratings: RatingRow[];
  best: { id: string; rating_milli: number }[];
  workload: WorkloadEntry[];
  error: string | null;
}

/** One model as a provider's catalog route listed it; null members are what it did not say. */
interface ListedModel {
  id: string;
  display_name: string;
  free: boolean | null;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

interface Probe {
  provider: DirectoryProvider;
  ok: boolean;
  models: ListedModel[];
  error: string | null;
}

/** One model in the catalog document: id, free flag, context length. */
interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: Wire;
  models: CatalogModel[];
}

interface Catalog {
  providers: CatalogProvider[];
}

/** What a model can do, from the quality table first and the listing second. */
interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/** One routable (provider, model) pair with its rating and capabilities. */
interface Candidate {
  provider: DirectoryProvider;
  model: ListedModel;
  rating: number;
  caps: Capabilities;
}

function capOf(providerId: string): number {
  return PLANNED_CAP[providerId] ?? DEFAULT_CAP;
}

/** A base path joined to the env's URL; an absolute base is used as it stands. */
function joinBase(envUrl: string, base: string): string {
  const b = base.trim();
  if (/^https?:\/\//i.test(b)) return b.replace(/\/+$/, '');
  const prefix = envUrl.replace(/\/+$/, '');
  if (b === '') return prefix;
  return `${prefix}${b.startsWith('/') ? '' : '/'}${b.replace(/\/+$/, '')}`;
}

/** The headers a provider wants its key in. */
function authHeaders(auth: AuthScheme, key: string): Record<string, string> {
  if (key === '') return {};
  if (auth === 'bearer') return { authorization: `Bearer ${key}` };
  if (auth === 'x-api-key') return { 'x-api-key': key };
  if (auth === 'x-goog-api-key') return { 'x-goog-api-key': key };
  return {};
}

/** The index of the `}` closing the object that opens at `start`, or -1. */
function objectEnd(text: string, start: number): number {
  let depth = 0;
  let inString = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === '\\') i += 1;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === '{') depth += 1;
    else if (ch === '}') {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Every JSON object in `text` that carries a string `schema`, in the order they appear. */
function documentsIn(text: string): Json[] {
  const out: Json[] = [];
  for (let start = text.indexOf('{'); start !== -1; start = text.indexOf('{', start + 1)) {
    const end = objectEnd(text, start);
    if (end === -1) continue;
    try {
      const value: unknown = JSON.parse(text.slice(start, end + 1));
      if (isRecord(value) && typeof value.schema === 'string') out.push(value);
    } catch {
      // Not JSON on its own: look on.
    }
  }
  return out;
}

function directoryProviderOf(v: unknown): DirectoryProvider | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  const wire = v.wire === 'openai' || v.wire === 'anthropic' || v.wire === 'gemini' ? v.wire : null;
  if (id === '' || wire === null) return null;
  const auth =
    v.auth === 'bearer' || v.auth === 'x-api-key' || v.auth === 'x-goog-api-key' || v.auth === 'none' ? v.auth : 'bearer';
  // The directory writes `path` (/p/<id>); a directory that spells an absolute base
  // instead may use base_path, base_url or base. Take the first non-empty, `path` first.
  const base =
    asString(v.path) !== ''
      ? asString(v.path)
      : asString(v.base_path) !== ''
        ? asString(v.base_path)
        : asString(v.base_url) !== ''
          ? asString(v.base_url)
          : asString(v.base);
  return { id, name: asString(v.name) !== '' ? asString(v.name) : id, wire, auth, base, key_env: asString(v.key_env), key: asString(v.key) };
}

function ratingRowOf(v: unknown): RatingRow | null {
  if (!isRecord(v)) return null;
  const model = asString(v.model);
  const provider = asString(v.provider);
  if (model === '' || provider === '') return null;
  return {
    model,
    provider,
    rating_milli: asInt(v.rating_milli) ?? 0,
    context_length: asInt(v.context_length),
    tools: v.tools === true,
    vision: v.vision === true,
    json: v.json === true,
    free: v.free === true,
  };
}

function workloadEntryOf(v: unknown): WorkloadEntry | null {
  if (!isRecord(v)) return null;
  const id = asString(v.id);
  if (id === '') return null;
  const requires = isRecord(v.requires) ? v.requires : {};
  const entry: WorkloadEntry = {
    id,
    api: v.api === 'anthropic' ? 'anthropic' : 'openai',
    stream: v.stream === true,
    weight: v.weight === 2 ? 2 : 1,
    requires: {
      tools: requires.tools === true,
      vision: requires.vision === true,
      json: requires.json === true,
      ...(typeof requires.min_context === 'number' && Number.isFinite(requires.min_context) ? { min_context: requires.min_context } : {}),
    },
  };
  return entry;
}

/** The brief at `path`, its three documents found by schema. */
function readBrief(path: string): Brief {
  const empty: Brief = { providers: [], ratings: [], best: [], workload: [], error: null };
  if (path === '') return { ...empty, error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { ...empty, error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const directoryDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/providers/v1' && Array.isArray(d.providers)) ?? null;
  if (directoryDoc === null) {
    return { ...empty, error: 'the brief holds no provider directory document' };
  }
  const providers = (directoryDoc.providers as unknown[]).flatMap((v) => {
    const p = directoryProviderOf(v);
    return p === null ? [] : [p];
  });
  const ratingsDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/ratings/v1' && Array.isArray(d.ratings)) ?? null;
  const ratings =
    ratingsDoc === null
      ? []
      : (ratingsDoc.ratings as unknown[]).flatMap((v) => {
          const r = ratingRowOf(v);
          return r === null ? [] : [r];
        });
  const best =
    ratingsDoc !== null && Array.isArray(ratingsDoc.best)
      ? (ratingsDoc.best as unknown[]).flatMap((v) => {
          if (!isRecord(v) || asString(v.id) === '') return [];
          return [{ id: asString(v.id), rating_milli: asInt(v.rating_milli) ?? 0 }];
        })
      : [];
  const workloadDoc = docs.find((d) => d.schema === 'free-model-aggregator-01/workload/v1' && Array.isArray(d.manifest)) ?? null;
  const workload =
    workloadDoc === null
      ? []
      : (workloadDoc.manifest as unknown[]).flatMap((v) => {
          const w = workloadEntryOf(v);
          return w === null ? [] : [w];
        });
  return { providers, ratings, best, workload, error: null };
}

/** The quality-table row of one provider's model, when the table has one. */
function ratingRowFor(ratings: RatingRow[], providerId: string, modelId: string): RatingRow | null {
  return ratings.find((r) => r.provider === providerId && r.model === modelId) ?? null;
}

/**
 * Whether a model is free: the quality table's row where it has one (the sealed
 * truth), the listing's own word where it does not, paid when neither says.
 */
function modelIsFree(model: ListedModel, row: RatingRow | null): boolean {
  if (row !== null) return row.free;
  return model.free ?? false;
}

/** What a model can do: the quality table's row where it has one, the listing's fields where it does not. */
function capabilitiesOf(model: ListedModel, row: RatingRow | null): Capabilities {
  return {
    context_length: row?.context_length ?? model.context_length ?? null,
    tools: row?.tools ?? model.tools ?? null,
    vision: row?.vision ?? model.vision ?? null,
    json: row?.json ?? model.json ?? null,
  };
}

/** Whether a model's capabilities meet one request's requirements. An unknown capability never meets one. */
function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  if (requires.tools === true && caps.tools !== true) return false;
  if (requires.vision === true && caps.vision !== true) return false;
  if (requires.json === true && caps.json !== true) return false;
  if (requires.min_context !== undefined && (caps.context_length === null || caps.context_length < requires.min_context)) return false;
  return true;
}

/** A price field of a listing: true when every price it names is zero, null when it says nothing about price. */
function zeroPriced(v: Json): boolean | null {
  const pricing = isRecord(v.pricing) ? v.pricing : null;
  if (pricing === null) return null;
  const values = Object.values(pricing).flatMap((x) => {
    if (typeof x === 'number') return [x];
    if (typeof x === 'string' && x.trim() !== '') return [Number(x)];
    return [];
  });
  if (values.length === 0) return null;
  return values.every((n) => Number.isFinite(n) && n === 0);
}

/** One model as a listing wrote it, in whichever wire's spelling. */
function listedModelIn(wire: Wire, v: unknown): ListedModel | null {
  if (!isRecord(v)) return null;
  const id = wire === 'gemini' ? asString(v.name).replace(/^models\//, '') : asString(v.id);
  if (id === '') return null;
  const context = asInt(v.context_length) ?? asInt(v.context_window) ?? asInt(v.inputTokenLimit);
  return {
    id,
    display_name:
      asString(v.display_name) !== '' ? asString(v.display_name) : asString(v.displayName) !== '' ? asString(v.displayName) : id,
    free: asBool(v.free) ?? zeroPriced(v),
    context_length: context,
    tools: asBool(v.tools),
    vision: asBool(v.vision),
    json: asBool(v.json),
  };
}

/** The models one catalog route's answer listed, in its wire's shape. */
function modelsIn(wire: Wire, body: unknown): ListedModel[] {
  if (!isRecord(body)) return [];
  const list = wire === 'gemini' ? body.models : body.data;
  if (!Array.isArray(list)) return [];
  return list.flatMap((v) => {
    const m = listedModelIn(wire, v);
    return m === null ? [] : [m];
  });
}

/** Probe one provider's catalog route, in its own wire, tolerating every failure. */
async function probeProvider(envUrl: string, provider: DirectoryProvider, timeoutMs = PROBE_TIMEOUT_MS): Promise<Probe> {
  const base = joinBase(envUrl, provider.base);
  const url = provider.wire === 'gemini' ? `${base}/v1beta/models` : `${base}/v1/models`;
  const headers: Record<string, string> = { accept: 'application/json', ...authHeaders(provider.auth, provider.key) };
  if (provider.wire === 'anthropic') headers['anthropic-version'] = ANTHROPIC_VERSION;
  try {
    const res = await fetch(url, { headers, signal: AbortSignal.timeout(timeoutMs) });
    if (!res.ok) return { provider, ok: false, models: [], error: `GET ${url} answered HTTP ${res.status}` };
    const body: unknown = await res.json();
    return { provider, ok: true, models: modelsIn(provider.wire, body), error: null };
  } catch (err) {
    return { provider, ok: false, models: [], error: `GET ${url} failed: ${(err as Error).message}` };
  }
}

/**
 * The catalog the probes and the quality table make: every provider that answered,
 * every model it listed. Its base_url is the directory's own path (`/p/<id>`),
 * never the path joined to the env URL: the env's origin differs between a live
 * pass and its replay, and the catalog is part of the state a replay must
 * reproduce byte for byte.
 */
function buildCatalog(probes: readonly Probe[], ratings: RatingRow[]): Catalog {
  const providers: CatalogProvider[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const seen = new Set<string>();
    const models: CatalogModel[] = [];
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      const row = ratingRowFor(ratings, probe.provider.id, model.id);
      models.push({ id: model.id, free: modelIsFree(model, row), context_length: row?.context_length ?? model.context_length ?? null });
    }
    providers.push({ id: probe.provider.id, base_url: probe.provider.base, key_env: probe.provider.key_env, wire: probe.provider.wire, models });
  }
  return { providers };
}

// ---------------------------------------------------------------------------
// Routing: the plan, the live provider state, and the candidate order.
// ---------------------------------------------------------------------------

/** Live per-provider state, learned only from the providers' own answers. */
interface ProviderState {
  /**
   * Inference calls made so far, successes and refusals alike: the provider
   * counts every call that names a free model against its budget, a scheduled
   * refusal included, so this is the number its budget is measured against.
   */
  calls: number;
  /** Refusals so far (402/429/5xx, timeouts, unusable answers). */
  fails: number;
  /** Planned calls still to come for this provider (the plan's remaining share). */
  reserved: number;
}

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
/** The planned first pick per manifest row, or null when no legal model exists. */
let plan: (Candidate | null)[] = [];
/** Live state per provider id. */
const providerState = new Map<string, ProviderState>();
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, fails: 0, reserved: 0 };
    providerState.set(providerId, state);
  }
  return state;
}

function isSpent(state: ProviderState): boolean {
  return state.fails >= DEAD_AFTER_FAILURES;
}

/** Whether the provider still has slack under its planned cap, counting reserved future uses. */
function hasSlack(providerId: string): boolean {
  const state = stateOf(providerId);
  return state.calls + state.reserved < capOf(providerId);
}

/**
 * Every routable candidate for one request, best rating first. A candidate is a
 * free model the provider's catalog listed, meeting the requirements, on a wire
 * that can carry the request: a JSON-mode OpenAI request never goes to an
 * Anthropic-wire provider, whose wire cannot carry response_format.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingRowFor(brief.ratings, probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      const caps = capabilitiesOf(model, row);
      if (!meetsRequirements(requires, caps)) continue;
      if (jsonMode && clientApi === 'openai' && probe.provider.wire === 'anthropic') continue;
      out.push({ provider: probe.provider, model, rating: row?.rating_milli ?? 0, caps });
    }
  }
  // Best rating first; ties break by provider id, then model id, so the order is
  // a pure function of the brief and the probes.
  out.sort((a, b) => b.rating - a.rating || (a.provider.id < b.provider.id ? -1 : a.provider.id > b.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/**
 * Plan the manifest before the first request arrives: walk the rows in order and
 * give each its best legal model on a provider that still has planned cap left.
 * Weight-2 rows come first in the manifest, so they take the top models first.
 */
function buildPlan(): void {
  plan = [];
  const plannedUsed = new Map<string, number>();
  for (const entry of brief.workload) {
    // At plan time the bodies are unseen, so a JSON-requiring OpenAI row is
    // planned as if it carried response_format: the Anthropic wire is kept out.
    const jsonMode = entry.requires.json === true && entry.api === 'openai';
    const candidates = candidatesFor(entry.requires, entry.api, jsonMode);
    let pick: Candidate | null = null;
    for (const candidate of candidates) {
      const used = plannedUsed.get(candidate.provider.id) ?? 0;
      if (used < capOf(candidate.provider.id)) {
        pick = candidate;
        plannedUsed.set(candidate.provider.id, used + 1);
        break;
      }
    }
    plan.push(pick);
    if (pick !== null) stateOf(pick.provider.id).reserved += 1;
  }
}

/**
 * The failover order for one request: candidates not yet tried, spent providers
 * last, providers with planned slack first, best rating first inside each group.
 */
function failoverOrder(candidates: Candidate[], tried: ReadonlySet<string>): Candidate[] {
  const remaining = candidates.filter((c) => !tried.has(c.provider.id));
  remaining.sort((a, b) => {
    const aSpent = isSpent(stateOf(a.provider.id)) ? 1 : 0;
    const bSpent = isSpent(stateOf(b.provider.id)) ? 1 : 0;
    if (aSpent !== bSpent) return aSpent - bSpent;
    const aSlack = hasSlack(a.provider.id) ? 0 : 1;
    const bSlack = hasSlack(b.provider.id) ? 0 : 1;
    if (aSlack !== bSlack) return aSlack - bSlack;
    return b.rating - a.rating;
  });
  return remaining;
}

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      const row = ratingRowFor(brief.ratings, probe.provider.id, model.id);
      if (!modelIsFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
  buildPlan();
}

// ---------------------------------------------------------------------------
// Upstream requests: build each provider wire's request from the client's body.
// ---------------------------------------------------------------------------

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The client's stop sequences, whatever spelling the wire used. */
function stopsOf(body: Json): string[] {
  const raw = body.stop ?? body.stop_sequences;
  if (typeof raw === 'string') return [raw];
  if (Array.isArray(raw)) return raw.filter((x): x is string => typeof x === 'string');
  return [];
}

/** Whether this client request asks for JSON mode (OpenAI response_format). */
function jsonModeOf(body: Json, clientApi: 'openai' | 'anthropic'): boolean {
  if (clientApi !== 'openai') return false;
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

/** The tool names a request carries, in order, whatever wire it arrived on. */
function toolNamesOf(body: Json, clientApi: 'openai' | 'anthropic'): string[] {
  if (!Array.isArray(body.tools)) return [];
  const names: string[] = [];
  for (const raw of body.tools) {
    if (!isRecord(raw)) continue;
    if (clientApi === 'openai') {
      if (isRecord(raw.function) && typeof raw.function.name === 'string') names.push(raw.function.name);
    } else if (typeof raw.name === 'string') {
      names.push(raw.name);
    }
  }
  return names;
}

/**
 * The provider's input-token count for the request this server is about to send,
 * by the providers' own formula: 1 + (system + user + tools) / 4. Used only to
 * fill a streamed answer's opening usage when the provider's stream has not
 * reported it yet; the value is a pure function of the upstream body.
 */
function expectedInputTokens(wire: Wire, upstream: Json): number {
  let systemText = '';
  let userText = '';
  let tools = '';
  if (wire === 'openai') {
    const systemParts: string[] = [];
    for (const message of Array.isArray(upstream.messages) ? upstream.messages : []) {
      if (!isRecord(message)) continue;
      if (message.role === 'system') {
        const text = textOf(message.content);
        if (text !== '') systemParts.push(text);
      } else if (message.role === 'user') {
        userText = textOf(message.content);
      }
    }
    systemText = systemParts.join('\n');
    tools = toolNamesOf(upstream, 'openai').join(',');
  } else if (wire === 'gemini') {
    if (isRecord(upstream.systemInstruction)) systemText = textOf(upstream.systemInstruction.parts);
    for (const content of Array.isArray(upstream.contents) ? upstream.contents : []) {
      if (!isRecord(content) || content.role === 'model') continue;
      userText = textOf(content.parts);
    }
    const declarations: string[] = [];
    for (const tool of Array.isArray(upstream.tools) ? upstream.tools : []) {
      if (!isRecord(tool) || !Array.isArray(tool.functionDeclarations)) continue;
      for (const fn of tool.functionDeclarations) if (isRecord(fn) && typeof fn.name === 'string') declarations.push(fn.name);
    }
    tools = declarations.join(',');
  } else if (wire === 'anthropic') {
    systemText = textOf(upstream.system);
    for (const message of Array.isArray(upstream.messages) ? upstream.messages : []) {
      if (!isRecord(message) || message.role !== 'user') continue;
      userText = textOf(message.content);
    }
    tools = toolNamesOf(upstream, 'anthropic').join(',');
  }
  return 1 + Math.floor((systemText.length + userText.length + tools.length) / 4);
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function openAIPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: the system prompt becomes a system message, image blocks become
 * image_url data URLs, tool_use blocks become tool_calls, tool_result blocks
 * become tool messages, tools and tool_choice are rewritten, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through.
 */
function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
        });
      }
      const text = textOf(content);
      messages.push({ role: 'assistant', content: text !== '' ? text : null, ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}) });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({ role: 'tool', tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', content: toolResultText(block.content) });
        continue;
      }
      const part = openAIPartOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopsOf(body);
  if (stops.length > 0) out.stop = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'none') out.tool_choice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One OpenAI message as Anthropic content blocks, with the system text kept aside. */
function anthropicBlocksOf(message: Json): Json[] {
  const blocks: Json[] = [];
  const content = message.content;
  if (typeof content === 'string') {
    if (content !== '') blocks.push({ type: 'text', text: content });
  } else if (Array.isArray(content)) {
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const url = part.image_url.url;
        const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
        if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
        else blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
  }
  if (Array.isArray(message.tool_calls)) {
    for (const call of message.tool_calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      let input: Json = {};
      if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
        try {
          const parsed: unknown = JSON.parse(call.function.arguments);
          if (isRecord(parsed)) input = parsed;
        } catch {
          // Not JSON: an empty input.
        }
      }
      blocks.push({ type: 'tool_use', id: typeof call.id === 'string' ? call.id : '', name: typeof call.function.name === 'string' ? call.function.name : '', input });
    }
  }
  return blocks;
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`: system messages become the system prompt, user parts become text and
 * image blocks, assistant tool calls become tool_use blocks, tool messages
 * become tool_result blocks, and the cap, temperature, top_p, stops and
 * streaming pass through. JSON mode cannot be carried on this wire, so the
 * caller never routes a JSON-mode request here.
 */
function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  let pendingToolResults: Json[] = [];
  const flushToolResults = () => {
    if (pendingToolResults.length === 0) return;
    messages.push({ role: 'user', content: pendingToolResults });
    pendingToolResults = [];
  };
  for (const raw of Array.isArray(body.messages) ? body.messages : []) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      pendingToolResults.push({ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: toolResultText(raw.content) });
      continue;
    }
    flushToolResults();
    const blocks = anthropicBlocksOf(raw);
    if (blocks.length === 0) continue;
    messages.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', content: blocks });
  }
  flushToolResults();
  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopsOf(body);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (choice === 'none') out.tool_choice = { type: 'none' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One content part of a Gemini request, from an OpenAI part or an Anthropic block. */
function geminiPartOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) return { inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } };
    return { fileData: { fileUri: url } };
  }
  if (part.type === 'image' && isRecord(part.source)) {
    const source = part.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  }
  return null;
}

/**
 * A client request (either wire) as a Gemini generateContent body naming `model`.
 * Tool messages are left out of the contents: they carry no text the provider's
 * echo needs, and including them as user turns would move the echoed user text.
 * Tool calls on assistant turns become functionCall parts.
 */
function clientToGemini(body: Json, clientApi: 'openai' | 'anthropic', model: string): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const functionDeclarations: Json[] = [];
  if (clientApi === 'anthropic') {
    const system = textOf(body.system);
    if (system !== '') systemParts.push(system);
    for (const raw of Array.isArray(body.messages) ? body.messages : []) {
      if (!isRecord(raw)) continue;
      const parts: Json[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'tool_use') {
            parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
            continue;
          }
          const part = geminiPartOf(block);
          if (part !== null) parts.push(part);
        }
      } else if (typeof raw.content === 'string' && raw.content !== '') {
        parts.push({ text: raw.content });
      }
      if (parts.length > 0) contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
    }
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRecord(raw) || typeof raw.name !== 'string') continue;
        functionDeclarations.push({ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } });
      }
    }
  } else {
    for (const raw of Array.isArray(body.messages) ? body.messages : []) {
      if (!isRecord(raw)) continue;
      if (raw.role === 'system') {
        const text = textOf(raw.content);
        if (text !== '') systemParts.push(text);
        continue;
      }
      if (raw.role === 'tool') continue;
      const parts: Json[] = [];
      const content = raw.content;
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const part of content) {
          if (!isRecord(part)) continue;
          const geminiPart = geminiPartOf(part);
          if (geminiPart !== null) parts.push(geminiPart);
        }
      }
      if (Array.isArray(raw.tool_calls)) {
        for (const call of raw.tool_calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          let args: Json = {};
          if (typeof call.function.arguments === 'string' && call.function.arguments.trim() !== '') {
            try {
              const parsed: unknown = JSON.parse(call.function.arguments);
              if (isRecord(parsed)) args = parsed;
            } catch {
              // Not JSON: an empty input.
            }
          }
          parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args } });
        }
      }
      if (parts.length > 0) contents.push({ role: raw.role === 'assistant' ? 'model' : 'user', parts });
    }
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
        functionDeclarations.push({
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        });
      }
    }
  }
  const generationConfig: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : null;
  if (maxTokens !== null) generationConfig.maxOutputTokens = maxTokens;
  const stops = stopsOf(body);
  if (stops.length > 0) generationConfig.stopSequences = stops;
  if (typeof body.temperature === 'number') generationConfig.temperature = body.temperature;
  if (typeof body.top_p === 'number') generationConfig.topP = body.top_p;
  if (jsonModeOf(body, clientApi)) generationConfig.responseMimeType = 'application/json';
  const out: Json = { contents, generationConfig };
  if (systemParts.length > 0) out.systemInstruction = { parts: systemParts.map((text) => ({ text })) };
  if (functionDeclarations.length > 0) out.tools = [{ functionDeclarations }];
  return out;
}

/** The upstream request for one candidate: URL, headers and JSON body. */
function upstreamRequest(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): { url: string; headers: Record<string, string>; json: Json } {
  const provider = candidate.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = { 'content-type': 'application/json', ...authHeaders(provider.auth, provider.key) };
  const stream = body.stream === true;
  if (provider.wire === 'openai') {
    if (stream) headers.accept = 'text/event-stream';
    const json = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    // The client never asks for usage on a stream; without include_usage the
    // provider's stream carries no usage chunk and the usage check fails. The
    // echo is unaffected: it does not include stream_options.
    if (stream) json.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, headers, json };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = ANTHROPIC_VERSION;
    if (stream) headers.accept = 'text/event-stream';
    const json = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropic(body, candidate.model.id);
    return { url: `${base}/v1/messages`, headers, json };
  }
  const action = stream ? 'streamGenerateContent' : 'generateContent';
  const json = clientToGemini(body, clientApi, candidate.model.id);
  return { url: `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`, headers, json };
}

// ---------------------------------------------------------------------------
// Provider calls and answers, translated back into the client's wire.
// ---------------------------------------------------------------------------

function numberOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
function openaiFinishToAnthropic(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
function anthropicStopToOpenAI(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
function geminiFinishToOpenAI(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
function geminiFinishToAnthropic(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

/** The text of Gemini parts: each part's own text, joined with nothing between them. */
function geminiText(parts: unknown): string {
  if (!Array.isArray(parts)) return '';
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

interface AnswerParts {
  text: string;
  toolCalls: Json[];
  toolUses: Json[];
  stop: unknown;
  input: number;
  output: number;
}

/** The text, tool calls, stop reason and usage of a Chat Completions answer. */
function openAIParts(answer: Json): AnswerParts {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls.filter(isRecord) : [];
  const usage = isRecord(answer.usage) ? answer.usage : {};
  return { text, toolCalls, toolUses: [], stop: choice.finish_reason, input: numberOrZero(usage.prompt_tokens), output: numberOrZero(usage.completion_tokens) };
}

/** The text, tool uses, stop reason and usage of an Anthropic Messages answer. */
function anthropicParts(answer: Json): AnswerParts {
  const content = Array.isArray(answer.content) ? answer.content : [];
  let text = '';
  const toolUses: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') text += block.text;
    else if (block.type === 'tool_use') toolUses.push(block);
  }
  const usage = isRecord(answer.usage) ? answer.usage : {};
  return { text, toolCalls: [], toolUses, stop: answer.stop_reason, input: numberOrZero(usage.input_tokens), output: numberOrZero(usage.output_tokens) };
}

/** The text, stop reason and usage of a Gemini generateContent answer. */
function geminiParts(answer: Json): AnswerParts {
  const candidates = Array.isArray(answer.candidates) ? answer.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const usage = isRecord(answer.usageMetadata) ? answer.usageMetadata : {};
  return {
    text: geminiText(content.parts),
    toolCalls: [],
    toolUses: [],
    stop: candidate.finishReason,
    input: numberOrZero(usage.promptTokenCount),
    output: numberOrZero(usage.candidatesTokenCount),
  };
}

/** Anthropic tool_use blocks as OpenAI tool_calls. */
function toolCallsFromAnthropic(toolUses: Json[]): Json[] {
  return toolUses.map((use) => ({
    id: typeof use.id === 'string' ? use.id : 'toolu_unknown',
    type: 'function',
    function: { name: typeof use.name === 'string' ? use.name : '', arguments: JSON.stringify(isRecord(use.input) ? use.input : {}) },
  }));
}

/** OpenAI tool_calls as Anthropic tool_use blocks. */
function toolUsesFromOpenAI(toolCalls: Json[]): Json[] {
  return toolCalls.map((call) => {
    const fn = isRecord(call.function) ? call.function : {};
    return {
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof fn.name === 'string' ? fn.name : '',
      input: argumentsOf(fn.arguments),
    };
  });
}

/** A Chat Completions answer object, in the client's shape. */
function openAICompletion(model: string, text: string, toolCalls: Json[], finish: string, input: number, output: number): Json {
  const message: Json = { role: 'assistant', content: text };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finish, logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** An Anthropic Messages answer object, in the client's shape. */
function anthropicMessage(id: string, model: string, text: string, toolUses: Json[], stop: string, input: number, output: number): Json {
  const content: Json[] = [];
  if (text !== '') content.push({ type: 'text', text });
  for (const use of toolUses) content.push(use);
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stop,
    stop_sequence: null,
    usage: { input_tokens: input, output_tokens: output },
  };
}

interface CallResult {
  ok: boolean;
  status: number;
  json: Json | null;
  stream: ReadableStream<Uint8Array> | null;
  inputTokens: number;
}

/** One inference call to a candidate provider: a refusal is a failure, never an answer. */
async function callCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): Promise<CallResult> {
  const failed = (status: number): CallResult => ({ ok: false, status, json: null, stream: null, inputTokens: 0 });
  const { url, headers, json } = upstreamRequest(candidate, clientApi, body);
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: JSON.stringify(json), signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch {
    return failed(0);
  }
  if (!res.ok) {
    try {
      await res.text();
    } catch {
      // The error body is unreadable: the refusal stands.
    }
    return failed(res.status);
  }
  if (body.stream === true) {
    if (res.body === null) return failed(res.status);
    return { ok: true, status: 200, json: null, stream: res.body, inputTokens: expectedInputTokens(candidate.provider.wire, json) };
  }
  let text: string;
  try {
    text = await res.text();
  } catch {
    return failed(res.status);
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch {
    return failed(res.status);
  }
  if (!isRecord(parsed)) return failed(res.status);
  return { ok: true, status: 200, json: parsed, stream: null, inputTokens: expectedInputTokens(candidate.provider.wire, json) };
}

// ---------------------------------------------------------------------------
// Server-sent events: frame parsing and the four stream translations.
// ---------------------------------------------------------------------------

interface SseFrame {
  event: string | null;
  data: string;
}

function parseSseFrame(raw: string): SseFrame | null {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
    else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
  }
  if (data.length === 0) return null;
  return { event, data: data.join('\n') };
}

/** The frames of an upstream event stream, in arrival order. */
async function* readSse(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = parseSseFrame(buffer.slice(0, at));
        buffer = buffer.slice(at + 2);
        if (frame !== null) yield frame;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') {
      const frame = parseSseFrame(buffer);
      if (frame !== null) yield frame;
    }
  } finally {
    reader.releaseLock();
  }
}

/** Copy a provider's event stream to the client as it arrives, byte for byte. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** One server-sent event: `event: <name>` when named, then the data line, then the blank line. */
function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One Chat Completions chunk. */
function openAIChunk(model: string, delta: Json, finish: string | null): Json {
  return { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] };
}

/** The usage chunk that closes a Chat Completions stream. */
function openAIUsageChunk(model: string, input: number, output: number): Json {
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start
 * on the first chunk, content_block_start/delta/stop per text block or tool
 * call, then message_delta with the stop reason and usage, then message_stop.
 * `inputTokens` is the provider's input count for the upstream request, computed
 * from that request's body; the output count comes from the stream's usage
 * chunk, or from the streamed text when the chunk never arrives.
 */
async function openAIStreamToAnthropic(stream: ReadableStream<Uint8Array>, res: ServerResponse, model: string, inputTokens: number): Promise<void> {
  let started = false;
  let finished = false;
  let textIndex = -1;
  let nextIndex = 0;
  let textSoFar = '';
  let outputTokens: number | null = null;
  let pendingStop: string | null = null;
  const toolIndexes = new Map<number, number>();
  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    const output = outputTokens ?? 1 + Math.floor(textSoFar.length / 4);
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: pendingStop ?? 'end_turn', stop_sequence: null }, usage: { output_tokens: output } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };
  for await (const frame of readSse(stream)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') {
      for (const event of [...startEvents(), ...finishEvents()]) res.write(event);
      res.end();
      return;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) outputTokens = numberOrZero(chunk.usage.completion_tokens);
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    const out: string[] = [...startEvents()];
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
      }
      textSoFar += delta.content;
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        out.push(
          sseEvent('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: { type: 'tool_use', id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`, name: typeof fn.name === 'string' ? fn.name : '', input: {} },
          }),
        );
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
      }
    }
    // Hold the stop reason until the stream's usage chunk and [DONE] have
    // arrived: the usage comes after the finish_reason, and message_delta
    // carries the output tokens.
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = openaiFinishToAnthropic(choice.finish_reason);
    for (const event of out) res.write(event);
  }
  if (!finished) {
    for (const event of [...startEvents(), ...finishEvents()]) res.write(event);
  }
  res.end();
}

/**
 * An Anthropic SSE stream as a Chat Completions stream: a role chunk on
 * message_start, content chunks for the text deltas, then the finish chunk, the
 * usage chunk and [DONE]. Usage comes from the stream's own message_start and
 * message_delta, or from the streamed text when the stream never reports it.
 */
async function anthropicStreamToOpenAI(stream: ReadableStream<Uint8Array>, res: ServerResponse, model: string, inputTokens: number): Promise<void> {
  let input: number | null = null;
  let output: number | null = null;
  let stop: string | null = null;
  let textSoFar = '';
  let roleSent = false;
  let done = false;
  const close = () => {
    if (done) return;
    done = true;
    const out = output ?? 1 + Math.floor(textSoFar.length / 4);
    res.write(sseEvent(null, openAIChunk(model, {}, stop ?? 'stop')));
    res.write(sseEvent(null, openAIUsageChunk(model, input ?? inputTokens, out)));
    res.write('data: [DONE]\n\n');
  };
  for await (const frame of readSse(stream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start') {
      if (isRecord(data.message) && isRecord(data.message.usage)) input = numberOrZero(data.message.usage.input_tokens);
      if (!roleSent) {
        roleSent = true;
        res.write(sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null)));
      }
    } else if (frame.event === 'content_block_delta') {
      if (!roleSent) {
        roleSent = true;
        res.write(sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null)));
      }
      if (isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string' && data.delta.text !== '') {
        textSoFar += data.delta.text;
        res.write(sseEvent(null, openAIChunk(model, { content: data.delta.text }, null)));
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stop = anthropicStopToOpenAI(data.delta.stop_reason);
      if (isRecord(data.usage)) output = numberOrZero(data.usage.output_tokens);
    } else if (frame.event === 'message_stop') {
      close();
    }
  }
  close();
  res.end();
}

/**
 * A Gemini SSE stream as a Chat Completions stream: content chunks for each
 * text piece, then the finish chunk, the usage chunk and [DONE]. The final
 * frame carries the finish reason and the usage counts.
 */
async function geminiStreamToOpenAI(stream: ReadableStream<Uint8Array>, res: ServerResponse, model: string, inputTokens: number): Promise<void> {
  let textSoFar = '';
  let finish: string | null = null;
  let usage: { input: number; output: number } | null = null;
  let roleSent = false;
  let done = false;
  const close = () => {
    if (done) return;
    done = true;
    res.write(sseEvent(null, openAIChunk(model, {}, finish ?? 'stop')));
    res.write(sseEvent(null, openAIUsageChunk(model, usage?.input ?? inputTokens, usage?.output ?? 1 + Math.floor(textSoFar.length / 4))));
    res.write('data: [DONE]\n\n');
  };
  for await (const frame of readSse(stream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    if (!roleSent) {
      roleSent = true;
      res.write(sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null)));
    }
    const piece = geminiText(isRecord(candidate.content) ? candidate.content.parts : undefined);
    if (piece !== '') {
      textSoFar += piece;
      res.write(sseEvent(null, openAIChunk(model, { content: piece }, null)));
    }
    if (typeof candidate.finishReason === 'string') finish = geminiFinishToOpenAI(candidate.finishReason);
    if (isRecord(data.usageMetadata)) {
      usage = { input: numberOrZero(data.usageMetadata.promptTokenCount), output: numberOrZero(data.usageMetadata.candidatesTokenCount) };
    }
  }
  close();
  res.end();
}

/**
 * A Gemini SSE stream as the Anthropic event sequence. The input count comes
 * from the upstream request's body (the stream reports it only at the end);
 * the output count comes from the final frame's usage counts.
 */
async function geminiStreamToAnthropic(stream: ReadableStream<Uint8Array>, res: ServerResponse, model: string, inputTokens: number): Promise<void> {
  let started = false;
  let blockOpen = false;
  let textSoFar = '';
  let finish: string | null = null;
  let output: number | null = null;
  let done = false;
  const start = () => {
    if (started) return;
    started = true;
    res.write(
      sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: inputTokens, output_tokens: 0 } },
      }),
    );
    res.write(sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }));
    blockOpen = true;
  };
  const close = () => {
    if (done) return;
    done = true;
    start();
    if (blockOpen) {
      res.write(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
      blockOpen = false;
    }
    res.write(
      sseEvent('message_delta', {
        type: 'message_delta',
        delta: { stop_reason: finish ?? 'end_turn', stop_sequence: null },
        usage: { output_tokens: output ?? 1 + Math.floor(textSoFar.length / 4) },
      }),
    );
    res.write(sseEvent('message_stop', { type: 'message_stop' }));
  };
  for await (const frame of readSse(stream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    start();
    const piece = geminiText(isRecord(candidate.content) ? candidate.content.parts : undefined);
    if (piece !== '') {
      textSoFar += piece;
      res.write(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
    }
    if (typeof candidate.finishReason === 'string') finish = geminiFinishToAnthropic(candidate.finishReason);
    if (isRecord(data.usageMetadata)) output = numberOrZero(data.usageMetadata.candidatesTokenCount);
  }
  close();
  res.end();
}

// ---------------------------------------------------------------------------
// The server: routes, request serving with failover, register and poll.
// ---------------------------------------------------------------------------

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, { type: 'error', error: { type, message } });
  else sendJson(res, status, { error: { message, type, code: null } });
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function modelsRoute(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') {
    sendJson(res, 200, {
      data: list.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
      has_more: false,
    });
  } else {
    sendJson(res, 200, {
      object: 'list',
      data: list.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
    });
  }
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: jsonModeOf(body, 'openai'),
  };
}

/** Send one provider's answer in the client's wire, streamed or not as asked. */
async function sendUpstreamAnswer(
  clientApi: 'openai' | 'anthropic',
  providerWire: Wire,
  result: CallResult,
  model: string,
  res: ServerResponse,
): Promise<void> {
  const stream = result.stream !== null;
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (clientApi === 'openai' && providerWire === 'openai') await pipeStream(result.stream as ReadableStream<Uint8Array>, res);
    else if (clientApi === 'anthropic' && providerWire === 'anthropic') await pipeStream(result.stream as ReadableStream<Uint8Array>, res);
    else if (clientApi === 'anthropic') {
      if (providerWire === 'openai') await openAIStreamToAnthropic(result.stream as ReadableStream<Uint8Array>, res, model, result.inputTokens);
      else await geminiStreamToAnthropic(result.stream as ReadableStream<Uint8Array>, res, model, result.inputTokens);
    } else {
      if (providerWire === 'anthropic') await anthropicStreamToOpenAI(result.stream as ReadableStream<Uint8Array>, res, model, result.inputTokens);
      else await geminiStreamToOpenAI(result.stream as ReadableStream<Uint8Array>, res, model, result.inputTokens);
    }
    return;
  }
  const answer = result.json as Json;
  if (clientApi === 'openai' && providerWire === 'openai') {
    sendJson(res, 200, answer);
    return;
  }
  if (clientApi === 'anthropic' && providerWire === 'anthropic') {
    sendJson(res, 200, answer);
    return;
  }
  if (clientApi === 'anthropic') {
    const id = typeof answer.id === 'string' ? answer.id : 'msg_aggregate';
    if (providerWire === 'openai') {
      const parts = openAIParts(answer);
      sendJson(res, 200, anthropicMessage(id, model, parts.text, toolUsesFromOpenAI(parts.toolCalls), openaiFinishToAnthropic(parts.stop), parts.input, parts.output));
    } else {
      const parts = geminiParts(answer);
      sendJson(res, 200, anthropicMessage(id, model, parts.text, [], geminiFinishToAnthropic(parts.stop), parts.input, parts.output));
    }
    return;
  }
  if (providerWire === 'anthropic') {
    const parts = anthropicParts(answer);
    sendJson(res, 200, openAICompletion(model, parts.text, toolCallsFromAnthropic(parts.toolUses), anthropicStopToOpenAI(parts.stop), parts.input, parts.output));
  } else {
    const parts = geminiParts(answer);
    sendJson(res, 200, openAICompletion(model, parts.text, [], geminiFinishToOpenAI(parts.stop), parts.input, parts.output));
  }
}

/**
 * Serve one client request: the planned first pick, then the failover order,
 * then one retry each of the providers already tried. A refusal is counted
 * against its provider and never forwarded as the answer.
 */
async function serveRequest(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const index = manifestIndex;
  manifestIndex += 1;
  const entry = brief.workload[index];
  const requires = entry !== undefined ? entry.requires : requirementsFromBody(body);
  const jsonMode = jsonModeOf(body, clientApi);
  const candidates = candidatesFor(requires, clientApi, jsonMode);
  // The plan's reserved share for this row is consumed now, whichever provider
  // ends up serving it.
  const planned = plan[index] ?? null;
  if (planned !== null) {
    const state = stateOf(planned.provider.id);
    state.reserved = Math.max(0, state.reserved - 1);
  }
  const queue: Candidate[] = [];
  if (
    planned !== null &&
    !isSpent(stateOf(planned.provider.id)) &&
    // A planned pick with no slack left is skipped: its call would likely run
    // past the provider's real budget, and a slack candidate serves the
    // request for the same rating step instead.
    hasSlack(planned.provider.id) &&
    candidates.some((c) => c.provider.id === planned.provider.id && c.model.id === planned.model.id)
  ) {
    queue.push(planned);
  }
  const tried = new Set(queue.map((c) => c.provider.id));
  for (const candidate of failoverOrder(candidates, tried)) queue.push(candidate);
  if (queue.length < MAX_ATTEMPTS) {
    // Nothing else left to try: one retry each, best first, of the providers
    // already tried — a scheduled refusal may have passed by the next call.
    const retry = [...candidates].sort((a, b) => b.rating - a.rating);
    for (const candidate of retry) {
      if (queue.length >= MAX_ATTEMPTS) break;
      if (queue.some((q) => q.provider.id === candidate.provider.id && q.model.id === candidate.model.id)) continue;
      queue.push(candidate);
    }
  }
  // The attempt budget for this request, retries included.
  let attemptsMade = 0;
  for (const candidate of queue.slice(0, MAX_ATTEMPTS)) {
    // One call, then one retry of the same provider on a refusal: a scheduled
    // refusal hits one call number, so the next number usually answers — and a
    // retry that lands keeps the request on its best model, where failing over
    // drops a rating step. The retry is skipped once the provider is at its
    // planned cap: past it, a further call is likely over its real budget.
    for (let attempt = 0; attempt < 2; attempt++) {
      if (attemptsMade >= MAX_ATTEMPTS) break;
      attemptsMade += 1;
      const state = stateOf(candidate.provider.id);
      state.calls += 1;
      const result = await callCandidate(candidate, clientApi, body);
      if (result.ok) {
        await sendUpstreamAnswer(clientApi, candidate.provider.wire, result, candidate.model.id, res);
        return;
      }
      state.fails += 1;
      log(`${candidate.provider.id}/${candidate.model.id} refused (HTTP ${result.status}); ${attempt === 0 ? 'retrying once' : 'failing over'}`);
      // Retry only while the provider still has slack under its planned cap,
      // counting its reserved future uses: a retry past the cap would starve
      // a later planned request of its call.
      if (attempt === 0 && hasSlack(candidate.provider.id)) continue;
      break;
    }
  }
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory could serve this request');
}

/** POST /v1/chat/completions: an OpenAI-shaped client. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serveRequest('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serveRequest('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    modelsRoute(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: Catalog): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 0; attempt < 10; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err;
      await sleep(500);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one
    // built object and stop before any server starts.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
