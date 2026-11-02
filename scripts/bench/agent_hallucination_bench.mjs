#!/usr/bin/env node
/**
 * agent_hallucination_bench.mjs — parallel agentic-load hallucination bench
 * for the sglang Qwen3.8-Flash-Next deployment (testcomp2).
 *
 * WHY THIS SHAPE (from hallucination-investigation.md + proxy-log forensics):
 *  - The confirmed leakage class (phantom folders, cross-session tool names,
 *    e.g. the `kirill` / `c4b7c840` incidents) appeared under 6-8 parallel
 *    heavy-agent sessions with 100-200K-token contexts, saturating the KV pool
 *    (--max-running-requests 32, --max-total-tokens 1048576).
 *  - The race is in the HiCache host-tier load-back: a shared prefix is
 *    evicted, then reloaded while overlap-scheduling writes to the freed
 *    pages. So the bench (a) uses a LARGE SHARED PREFIX across all lanes
 *    (like the real tool-catalog/system prompt) so the radix cache shares and
 *    offloads it, (b) churns sessions (finish -> new session) so the prefix
 *    must be reloaded repeatedly, and (c) keeps >= 6-8 lanes busy so the
 *    reload overlaps with in-flight writes. That is the fence's race window.
 *  - Detection: every session carries a unique canary (8-hex + /work/<hex>/
 *    folders) present only in its own context. A canary appearing in another
 *    lane's output = cross-session leak. Unknown tool names in tool_calls =
 *    the "not defined in the tools list" class.
 *
 * Usage:
 *   node agent_hallucination_bench.mjs [--lanes 8] [--duration 15]
 *        [--base http://192.168.2.208:19999/v1] [--model qwen-3.8-flash-next]
 *        [--out bench-run.jsonl]
 *
 * Env overrides: BASE_URL, MODEL, LANES, DURATION (minutes), OUT.
 * Exit code: 1 if any leak or unknown-tool hit (usable as a CI gate).
 */

import { appendFileSync, writeFileSync } from 'node:fs';

// ---------- config ----------
const argv = process.argv.slice(2);
function arg(name, dflt) {
  const i = argv.indexOf('--' + name);
  return i >= 0 && argv[i + 1] ? argv[i + 1] : dflt;
}
const BASE = arg('base', process.env.BASE_URL || 'http://192.168.2.208:19999/v1');
const MODEL = arg('model', process.env.MODEL || 'qwen-3.8-flash-next');
const LANES = parseInt(arg('lanes', process.env.LANES || '8'), 10);
const DURATION_MIN = parseFloat(arg('duration', process.env.DURATION || '15'));
const OUT = arg('out', process.env.OUT || 'bench-run.jsonl');
const MAX_CTX_BYTES = 420_000;   // churn the session once context grows past this
const MAX_TURNS = 22;            // per session
const MAX_TOKENS = 1200;         // per response (reasoning model: needs room for tool calls)
const SUB = parseInt(arg('sub', process.env.SUB || '2'), 10); // chains per lane (main + subagents)
const REQ_TIMEOUT_MS = 300_000;

// ---------- canaries ----------
function hex8() {
  return Array.from({ length: 8 }, () => '0123456789abcdef'[(Math.random() * 16) | 0]).join('');
}
// pre-generate a pool of canaries so churned sessions get fresh ones
const canaryPool = new Set();
function newCanary() {
  for (;;) {
    const h = hex8();
    if (!canaryPool.has(h)) { canaryPool.add(h); return h; }
  }
}

// ---------- shared prefix (identical for ALL lanes -> radix-shared + offloaded) ----------
// ~30KB, modeled on the real agent system prompt (tool catalog + rules).
const SYSTEM_PROMPT = [
  'You are Orchestrator, a workflow manager who plans, delegates, monitors,',
  'reconciles, and verifies specialist work on a shared build server.',
  'The server hosts many concurrent agent sessions. Treat every path as scoped',
  'to your current working directory. Never assume a directory exists: verify',
  'with the tools before reading or writing.',
  '',
  '## Tool catalog (abridged reference)',
  ...Array.from({ length: 40 }, (_, i) =>
    `${i + 1}. tool_rule_${i}: when using capability ${i}, always check the result code, never fabricate outputs, and keep arguments consistent with the declared schema.`),
  '',
  '## Style rules',
  ...Array.from({ length: 60 }, (_, i) =>
    `Rule ${i + 1}: be concise, no preamble, no flattery; state assumptions; verify claims before asserting them; prefer minimal diffs; run the narrowest check that proves the change; escalate only blocking decisions.`),
  '',
  '## Environment notes',
  'Workspace: /srv/agents (shared). Each session works in its own /work/<id>/ tree.',
  'The build server is a Linux box with 8 GPUs; heavy parallel load is normal.',
  'Retracted or finished sessions free their pages; the scheduler may reuse them.',
  'A hierarchical KV cache may reload old prefixes from the host tier on demand.',
  'Be aware that other sessions run concurrently; do not reference their paths.',
].join('\n');

// ---------- tool set (subset of the real deployment, correct param names) ----------
const TOOLS = [
  { type: 'function', function: { name: 'bash', description: 'Run a shell command.', parameters: { type: 'object', properties: { command: { type: 'string' } }, required: ['command'] } } },
  { type: 'function', function: { name: 'read', description: 'Read a file.', parameters: { type: 'object', properties: { path: { type: 'string' }, offset: { type: 'number' }, limit: { type: 'number' } }, required: ['path'] } } },
  { type: 'function', function: { name: 'write', description: 'Write a file.', parameters: { type: 'object', properties: { path: { type: 'string' }, content: { type: 'string' } }, required: ['path', 'content'] } } },
  { type: 'function', function: { name: 'edit', description: 'Edit a file.', parameters: { type: 'object', properties: { path: { type: 'string' }, oldString: { type: 'string' }, newString: { type: 'string' } }, required: ['path', 'oldString', 'newString'] } } },
  { type: 'function', function: { name: 'grep', description: 'Search file contents.', parameters: { type: 'object', properties: { pattern: { type: 'string' }, path: { type: 'string' } }, required: ['pattern'] } } },
  { type: 'function', function: { name: 'glob', description: 'Find files by glob.', parameters: { type: 'object', properties: { pattern: { type: 'string' }, path: { type: 'string' } }, required: ['pattern'] } } },
  { type: 'function', function: { name: 'task_status', description: 'Check a parallel task.', parameters: { type: 'object', properties: { sessionID: { type: 'string' } }, required: [] } } },
];
const TOOL_NAMES = new Set(TOOLS.map(t => t.function.name));
// decoys the model has been observed hallucinating (the known confabulation
// class) — used only for reporting, detection is "not in TOOL_NAMES"
const KNOWN_CONFABULATIONS = new Set(['exec', 'summary', 'Read', 'task_update', 'search', 'print', 'ask_user', 'canary.exe']);

// ---------- synthetic tool results (big, lane-scoped, canary-bearing) ----------
function syntheticToolResult(cwd, round) {
  const lines = [];
  lines.push(`$ ls -la ${cwd}/ && cat ${cwd}/notes.md`);
  lines.push(`total ${40 + round}`);
  lines.push(`drwxr-xr-x  4 agent agent 4096 .`);
  lines.push(`drwxr-xr-x 24 agent agent 4096 ..`);
  lines.push(`-rw-r--r--  1 agent agent 1234 round${round}.txt`);
  lines.push(`-rw-r--r--  1 agent agent 8888 state.json`);
  lines.push('--- notes.md ---');
  lines.push(`Session workdir: ${cwd}`);
  lines.push(`This session's token: ${cwd.split('/').pop()}`);
  lines.push(`Round ${round}: build artifacts staged under ${cwd}/build/ and ${cwd}/vendor/`);
  for (let i = 0; i < 26; i++) {
    lines.push(`${cwd}/build/artifact_${round}_${i}.o  ${1024 + i * 7} bytes  sha256=${hex8()}${hex8()}`);
  }
  // bulk filler to grow the context realistically (like real tool outputs)
  for (let i = 0; i < 120; i++) {
    lines.push(`[worker-${(i % 6) + 1}] ${cwd}/src/module_${i}.ts: compiled ${12 + (i % 30)} symbols, ${i % 5} warnings, hash ${hex8()}${hex8()}${hex8()}${hex8()}`);
  }
  return lines.join('\n');
}

// ---------- request plumbing ----------
async function chat(messages, deadline) {
  const body = {
    model: MODEL,
    messages,
    tools: TOOLS,
    stream: false,
    max_tokens: MAX_TOKENS,
    temperature: 0.7,
  };
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), Math.min(REQ_TIMEOUT_MS, deadline - Date.now()));
  const t0 = Date.now();
  try {
    const res = await fetch(`${BASE}/chat/completions`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
    const j = await res.json();
    return { msg: j.choices[0].message, usage: j.usage, ms: Date.now() - t0 };
  } finally {
    clearTimeout(timer);
  }
}

// ---------- leak detection ----------
// canaries: Map<hex8, laneIndex> (live + historical; historical keep for detection)
const canaryMap = new Map();
function scanText(text, laneIdx, where, out) {
  if (!text) return;
  for (const [hex, owner] of canaryMap) {
    if (owner !== laneIdx && text.includes(hex)) {
      out.push({ type: 'CROSS_SESSION_LEAK', canary: hex, owner, lane: laneIdx, where });
    }
  }
}
// ---------- novel-content detection (the incoherent-KV class) ----------
// The synthetic world is FULLY controlled: every legitimate path is
// /work/<hex>/... (the lane's own) or /srv/agents (from the system prompt).
// An IP address, a Windows drive path, or an absolute path outside that
// world is content the model cannot have seen — the 2026-11-01 HICACHE=1
// contamination class (192.168.10.100, /data/launchpad/, browser-laptop
// were born in responses, absent from every request in the proxy archive).
// The canary check alone missed this: it only sees verbatim copies of its
// own canaries. Novel content is the stronger detector.
//
// Whitelist (triaged 2026-11-02): generic single-segment OS paths that the
// model knows from TRAINING and deliberately probes with when it notices
// the bench's canned tool layer (/etc/hostname, /proc/1/..., /sys/...,
// /tmp/...). These are NOT session-specific, so they cannot evidence a
// cross-session leak — the incident class is always session/site-specific
// (LAN IPs, drive paths, /data/launchpad/, /home/<user>/, /c/Users/...).
// The /etc /proc /sys /tmp families are whitelisted; everything else in
// the list below stays flagged.
const NOVEL_ALLOW = /^\/(?:etc|proc|sys|tmp|dev|run|bin|sbin|lib|lib64|boot|media|cdrom|snap)(?:\/|$)/;
const NOVEL_RES = [
  { re: /\b(?:\d{1,3}\.){3}\d{1,3}\b/g, kind: 'ip' },
  { re: /\b[A-Za-z]:\\[^\s"'`)\]]+/g, kind: 'drive_path' },
  { re: /(?:^|[\s"'(])\/(?:data|home|root|Users|mnt|var|usr|opt|workspace|launchpad|storage)(?:\/|$)[^\s"'`)\]]*/g, kind: 'abs_path' },
  { re: /(?:^|[\s"'(])\/[a-z]\/[A-Za-z][^\s"'`)\]]*/g, kind: 'drive_path' },  // git-bash /c/Users/... (the kirill shape)
];
function scanNovel(text, laneIdx, where, out) {
  if (!text) return;
  for (const { re, kind } of NOVEL_RES) {
    re.lastIndex = 0;
    let m;
    while ((m = re.exec(text)) !== null) {
      const val = m[0].replace(/^[\s"'(]/, '').trim();
      // generic OS paths are model priors, not leaks (see NOVEL_ALLOW)
      if (kind === 'abs_path' && NOVEL_ALLOW.test(val)) continue;
      // localhost/loopback IPs are generic priors too (127.0.0.1, 0.0.0.0)
      if (kind === 'ip' && /^(?:127\.|0\.0\.0\.)/.test(val)) continue;
      out.push({ type: 'NOVEL_CONTENT', kind, value: val.slice(0, 80), lane: laneIdx, where });
      break;
    }
  }
}
function scanToolCalls(toolCalls, laneIdx, out) {
  for (const tc of toolCalls || []) {
    const name = tc.function?.name;
    if (name && !TOOL_NAMES.has(name)) {
      out.push({ type: 'UNKNOWN_TOOL', name, lane: laneIdx, known_confab: KNOWN_CONFABULATIONS.has(name) });
    }
    // tool arguments may also leak canaries or novel content
    scanText(tc.function?.arguments, laneIdx, 'tool_args', out);
    scanNovel(tc.function?.arguments, laneIdx, 'tool_args', out);
  }
}

// ---------- session driver ----------
function newSessionMessages(lane, reuseHex) {
  // A real subagent shares the orchestrator's canary/cwd; only a fresh
  // orchestrator session gets a new one.
  const hex = reuseHex || newCanary();
  if (!reuseHex) canaryMap.set(hex, lane.id);
  const cwd = `/work/${hex}`;
  if (!reuseHex) { lane.cwd = cwd; lane.hex = hex; }
  const msgs = [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: `You are agent session ${hex} in ${cwd}. Work through the checklist: build the project in ${cwd}, run the tests, and report failures. Keep working until the checklist is done; when you need information, use the tools. Start by listing the directory.` },
  ];
  return { msgs, hex };
}

async function runChain(lane, kind, deadline, log) {
  // kind: 'main' (big agentic loop) or 'sub' (smaller subagent-style task)
  let sessions = 0;
  while (Date.now() < deadline) {
    sessions++;
    if (kind === 'sub') {
      // wait for the main chain to establish this lane's canary/cwd
      while (!lane.hex && Date.now() < deadline) await sleep(500);
      if (!lane.hex) break;
    }
    const { msgs, hex } = newSessionMessages(lane, kind === 'sub' ? lane.hex : undefined);
    if (kind === 'sub') {
      msgs[1] = { role: 'user', content: `You are a subagent of session ${hex}. Read ${lane.cwd}/state.json and report the round count. Then run the tests in ${lane.cwd}/build. Keep it short.` };
    }
    const maxTurns = kind === 'sub' ? 10 : MAX_TURNS;
    const maxCtx = kind === 'sub' ? 120_000 : MAX_CTX_BYTES;
    let turns = 0;
    let ctxBytes = 0;
    while (turns < maxTurns && Date.now() < deadline && ctxBytes < maxCtx) {
      turns++;
      const hits = [];
      let resp;
      try {
        resp = await chat(msgs, deadline);
      } catch (e) {
        if (Date.now() >= deadline) {
          // the run's deadline cut this request in flight: clean shutdown,
          // not an error. Don't pollute the errors count.
          log({ ev: 'deadline', lane: lane.id, hex, kind, turn: turns });
          break;
        }
        log({ ev: 'error', lane: lane.id, hex, kind, turn: turns, err: String(e).slice(0, 200) });
        lane.errors++;
        await sleep(3000);
        continue;
      }
      const m = resp.msg;
      const textOut = (m.content || '') + '\n' + (m.reasoning_content || '');
      scanText(textOut, lane.id, 'output', hits);
      scanNovel(textOut, lane.id, 'output', hits);
      scanToolCalls(m.tool_calls, lane.id, hits);
      for (const h of hits) {
        lane.hits.push(h); lane.hitCount++;
        log({ ev: 'HIT', lane: lane.id, hex, kind, ...h });
        console.log(`!!! ${h.type} lane=${lane.id} ${h.name || h.canary || ''} (hex ${hex}) ${new Date().toISOString()}`);
      }
      ctxBytes += JSON.stringify(m).length;
      lane.reqs++;
      lane.tokIn += resp.usage?.prompt_tokens || 0;
      lane.tokOut += resp.usage?.completion_tokens || 0;
      log({
        ev: 'req', lane: lane.id, hex, kind, turn: turns, ms: resp.ms,
        in: resp.usage?.prompt_tokens, out: resp.usage?.completion_tokens,
        tool_calls: (m.tool_calls || []).map(t => t.function?.name).join(','),
        hits: hits.length,
      });
      msgs.push({ role: 'assistant', content: m.content || null, ...(m.tool_calls ? { tool_calls: m.tool_calls } : {}) });
      if (m.tool_calls && m.tool_calls.length) {
        for (const tc of m.tool_calls) {
          const tr = syntheticToolResult(lane.cwd, turns);
          msgs.push({ role: 'tool', tool_call_id: tc.id, content: tr });
          ctxBytes += tr.length;
        }
      } else {
        // no tool call: nudge to keep the loop agent-like
        msgs.push({ role: 'user', content: `Continue. Run the next checklist step in ${lane.cwd}.` });
      }
    }
    // session done -> churn (new session, same shared prefix -> forces
    // radix-shared prefix reload from the host tier under load)
  }
  return sessions;
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ---------- main ----------
const deadline = Date.now() + DURATION_MIN * 60_000;
const lanes = Array.from({ length: LANES }, (_, i) => ({
  id: i, reqs: 0, errors: 0, tokIn: 0, tokOut: 0, hits: [], hitCount: 0, sessions: 0,
}));
const log = (() => {
  let first = true;
  return (obj) => {
    const line = JSON.stringify({ t: Date.now(), ...obj });
    if (first) { writeFileSync(OUT, line + '\n'); first = false; }
    else appendFileSync(OUT, line + '\n');
  };
})();

console.log(`bench: ${LANES} lanes x ${DURATION_MIN} min -> ${BASE} (${MODEL})`);
console.log(`events -> ${OUT}`);
const t0 = Date.now();
const statusTimer = setInterval(() => {
  const done = lanes.reduce((a, l) => a + l.reqs, 0);
  const hits = lanes.reduce((a, l) => a + l.hitCount, 0);
  const errs = lanes.reduce((a, l) => a + l.errors, 0);
  const el = ((Date.now() - t0) / 1000).toFixed(0);
  console.log(`[${el}s] reqs=${done} leaks=${hits} errors=${errs}`);
}, 10000);

await Promise.all(lanes.map(async (l) => {
  const chains = [runChain(l, 'main', deadline, log)];
  for (let s = 1; s < SUB; s++) chains.push(runChain(l, 'sub', deadline, log));
  const done = await Promise.all(chains);
  l.sessions = done[0];
}));
clearInterval(statusTimer);

const reqs = lanes.reduce((a, l) => a + l.reqs, 0);
const errs = lanes.reduce((a, l) => a + l.errors, 0);
const allHits = lanes.flatMap(l => l.hits);
const leaks = allHits.filter(h => h.type === 'CROSS_SESSION_LEAK');
const unk = allHits.filter(h => h.type === 'UNKNOWN_TOOL');
const novel = allHits.filter(h => h.type === 'NOVEL_CONTENT');
const summary = {
  bench: 'agent_hallucination_bench',
  base: BASE, model: MODEL,
  duration_min: (Date.now() - t0) / 60000,
  lanes: LANES, sessions: lanes.reduce((a, l) => a + l.sessions, 0),
  reqs, errors: errs,
  tok_in: lanes.reduce((a, l) => a + l.tokIn, 0),
  tok_out: lanes.reduce((a, l) => a + l.tokOut, 0),
  cross_session_leaks: leaks.length,
  unknown_tool_calls: unk.length,
  unknown_tool_names: [...new Set(unk.map(h => h.name))],
  novel_content_hits: novel.length,
  novel_content_samples: novel.slice(0, 20),
  leaks: leaks.slice(0, 20),
};
writeFileSync(OUT.replace(/\.jsonl$/, '') + '.summary.json', JSON.stringify(summary, null, 2));
console.log(JSON.stringify(summary, null, 2));
process.exit(leaks.length + unk.length + novel.length > 0 ? 1 : 0);
