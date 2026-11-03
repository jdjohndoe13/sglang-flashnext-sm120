#!/usr/bin/env node
// coords_check_loop.mjs — verify a screenshot-measured click coordinate before
// clicking, via a marker-overlay adjustment loop with a direct model call.
//
// Why: the vision model's grounding can be systematically biased (the mrope
// kernel bug class) or simply noisy. The loop lets the model SEE where its
// candidate click lands on a fresh screenshot and self-correct, in the same
// coordinate space, before any real click happens.
//
// Design (verified against the live SGLang server + win-b VM, 2026-11-03):
//   - the judge is a DIRECT model call (POST <api>/chat/completions,
//     temperature 0, thinking off) — no OpenCode session exists at all, so
//     the marker images never enter any agent's context and nothing can be
//     resurrected by a server restart.
//   - each iteration: MCP-call the VM (capture + draw marker + GetPixel
//     self-check + base64 over the wire), send the marked image + the target
//     description to the model, parse the JSON verdict
//   - the model answers confirmed / adjust (corrected coords) / fail
//   - convergence proof: (1300,600) -> (283,27) -> (105,27) -> VERIFIED on
//     the VS Save button in 3 iterations, with the unpatched (biased) model
//
// Why NOT a forked OpenCode session: prompts sent to a fork are delivered to
// the PARENT session as user messages (verified 3x, 2026-11-03) — the exact
// context pollution this skill exists to avoid. The judge task is
// self-contained (image + target description), so inherited context adds
// nothing. (Forks are also top-level rows — projector.ts parent_id NULL —
// so the boot resume sweep could resurrect a crashed fork; the direct call
// has no such state.)
//
// Usage (positional args for the target — a quoted --target breaks under the
// portable shell scanner, which splits quoted args at the first space):
//   node coords_check_loop.mjs --x 1000 --y 400 [--vm win-b] [--max 3]
//     [--api http://192.168.2.208:19999/v1] [--model qwen-3.8-flash-next]
//     [--out <dir>] -- the Run button in the toolbar
//   (marked images are always saved to --out for post-hoc inspection)
//
// Exit codes: 0 = VERIFIED; 2 = UNVERIFIED (no convergence / bad answer);
//   3 = fail (transport error / missing args).
//
// Zero dependencies (node >= 18).

import { mkdirSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

// ---------- args ----------
const argv = process.argv.slice(2)
const flags = {}
const positional = []
let seenDashdash = false
for (let i = 0; i < argv.length; i++) {
  const a = argv[i]
  if (seenDashdash) { positional.push(a); continue }
  if (a === '--') { seenDashdash = true; continue }
  if (a.startsWith('--')) {
    const name = a.slice(2)
    if (name === 'keep') { flags.keep = true; continue }
    flags[name] = argv[i + 1] !== undefined && !argv[i + 1].startsWith('--') ? argv[++i] : true
  } else positional.push(a)
}
const arg = (n, d) => (flags[n] !== undefined ? flags[n] : d)
const num = (s, d) => {
  if (s === undefined) return d
  const n = Number(s)
  if (!Number.isInteger(n)) throw new Error('not an integer: ' + s)
  return n
}
const VM_NAMES = {
  'win-a': 'http://192.168.250.130:7777/mcp',
  'win-b': 'http://192.168.250.131:7777/mcp',
  'win-c': 'http://192.168.250.132:7777/mcp',
  'win-d': 'http://192.168.250.133:7777/mcp',
  'win-e': 'http://192.168.250.134:7777/mcp',
  'win-f': 'http://192.168.250.135:7777/mcp',
  'win-vmware': 'http://192.168.174.130:7777/mcp',
}
const VM = VM_NAMES[arg('vm', 'win-b')] || arg('vm', 'win-b')
const MAX = num(arg('max', '3'), 3)
let x = num(arg('x', ''), NaN)
let y = num(arg('y', ''), NaN)
const TARGET = positional.join(' ')
const API = String(arg('api', 'http://192.168.2.208:19999/v1')).replace(/\/+$/, '')
const MODEL = String(arg('model', 'qwen-3.8-flash-next'))
const OUT_DIR = arg('out', path.join(os.tmpdir(), 'coords_check'))
mkdirSync(OUT_DIR, { recursive: true })

const die = (code, msg) => {
  console.log('RESULT: ' + (code === 0 ? 'VERIFIED' : code === 2 ? 'UNVERIFIED' : 'FAIL') + ' — ' + msg)
  console.log('COORDS: ' + x + ' ' + y)
  process.exit(code)
}
if (!Number.isInteger(x) || !Number.isInteger(y)) die(3, 'missing/invalid --x/--y')
if (!TARGET) die(3, 'missing target description (trailing positional args)')

// ---------- VM MCP ----------
let mcpSid = null
let mcpN = 0
const mcp = async (method, params, timeoutMs = 120_000) => {
  const res = await fetch(VM, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json, text/event-stream',
      ...(mcpSid ? { 'Mcp-Session-Id': mcpSid } : {}),
    },
    body: JSON.stringify({ jsonrpc: '2.0', id: ++mcpN, method, params }),
    signal: AbortSignal.timeout(timeoutMs),
  })
  if (res.headers.get('mcp-session-id')) mcpSid = res.headers.get('mcp-session-id')
  const text = await res.text()
  let payload = null
  for (const line of text.split('\n'))
    if (line.startsWith('data:')) { try { payload = JSON.parse(line.slice(5).trim()) } catch {} }
  if (!payload && text.trim()) { try { payload = JSON.parse(text) } catch {} }
  if (!payload) throw new Error('MCP unparseable: ' + text.slice(0, 200))
  if (payload.error) throw new Error('MCP error: ' + JSON.stringify(payload.error).slice(0, 200))
  if (payload.result?.isError) throw new Error('MCP tool error: ' + JSON.stringify(payload.result).slice(0, 200))
  return payload.result
}
const mcpText = (r) => (r?.content || []).filter((c) => c.type === 'text').map((c) => c.text).join('')

// capture + draw + self-check; returns { b64, size, file }
const psCommand = (cx, cy) => `
$cx=${cx}; $cy=${cy}
if ($cx -lt 0 -or $cy -lt 0) { "OUTBOUNDS"; exit 1 }
Add-Type -AssemblyName System.Drawing,System.Windows.Forms
$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
if ($cx -ge $b.Width -or $cy -ge $b.Height) { "OUTBOUNDS"; exit 1 }
$r = 40
$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
$g.SmoothingMode = "AntiAlias"
$white = New-Object System.Drawing.Pen ([System.Drawing.Color]::White), 8
$blue = New-Object System.Drawing.Pen ([System.Drawing.Color]::FromArgb(255,0,120,255)), 4
$blue.DashStyle = "Dash"
$g.DrawEllipse($white, ($cx-$r), ($cy-$r), (2*$r), (2*$r))
$g.DrawEllipse($blue, ($cx-$r), ($cy-$r), (2*$r), (2*$r))
$g.DrawLine($white, ($cx-$r-10), $cy, ($cx-12), $cy)
$g.DrawLine($white, ($cx+12), $cy, ($cx+$r+10), $cy)
$g.DrawLine($white, $cx, ($cy-$r-10), $cx, ($cy-12))
$g.DrawLine($white, $cx, ($cy+12), $cx, ($cy+$r+10))
$out = "$env:TEMP\\coords_check_loop.png"
$bmp.Save($out, [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
$chk = New-Object System.Drawing.Bitmap $out
$ok = 0
# the ring is a blue DASH over an 8px white ring: a cardinal point is either
# blue (dash present) or white (dash gap). Accept both.
$tried = 0
foreach ($pt in @(@(($cx),($cy-$r)), @(($cx),($cy+$r)), @(($cx-$r),($cy)), @(($cx+$r),($cy)))) {
  if ($pt[0] -lt 0 -or $pt[1] -lt 0 -or $pt[0] -ge $b.Width -or $pt[1] -ge $b.Height) { continue }
  $tried++
  $p = $chk.GetPixel($pt[0], $pt[1])
  if (($p.B -gt 200 -and $p.R -gt 200 -and $p.G -gt 200) -or ($p.B -gt 200 -and $p.R -lt 80)) { $ok++ }
}
$chk.Dispose()
"SELFCHECK: ring $ok/$tried"
if ($tried -eq 0 -or $ok -lt $tried) { "MARKER_FAIL"; exit 1 }
"SCREENSIZE: $($b.Width) x $($b.Height)"
"MK" + [Convert]::ToBase64String([IO.File]::ReadAllBytes($out)) + "END"
Remove-Item $out
`
const captureMarked = async (cx, cy, step) => {
  const r = await mcp('tools/call', { name: 'PowerShell', arguments: { command: psCommand(cx, cy), timeout: 60 } })
  const text = mcpText(r)
  if (text.includes('OUTBOUNDS')) throw new Error('out of bounds: ' + cx + ',' + cy)
  if (text.includes('MARKER_FAIL')) throw new Error('marker self-check failed at (' + cx + ',' + cy + ')')
  const m = text.match(/MK([A-Za-z0-9+/=]+)END/s)
  if (!m) throw new Error('no base64 in VM response: ' + text.slice(0, 200))
  const size = (text.match(/SCREENSIZE: (\d+) x (\d+)/) || []).slice(1, 3).join('x')
  const file = path.join(OUT_DIR, `marked_${Date.now()}.png`)
  writeFileSync(file, Buffer.from(m[1], 'base64'))
  return { b64: m[1], size: size || 'unknown', file }
}

// ---------- the judge: direct model call, no session ----------
const verdictPrompt = (step, cx, cy, size) =>
  `COORDS CHECK — step ${step} of ${MAX}. The task needs to click: ${TARGET}.
The attached image is the VM desktop (${size}, full-screen pixel space) with a marker:
a blue dashed circle with a white outline and crosshair arms. The marker center is
EXACTLY the point that would be clicked — coordinates (${cx}, ${cy}) in full-screen space.

You are the same model that will do the click, so this is you checking your own
grounding. Look at the marker center and answer with ONE line of JSON:
- {"verdict":"confirmed","observed":"<what sits at the marker center>"} — the intended target is at/under the marker center (if the target is fully inside the ring, that counts as confirmed).
- {"verdict":"adjust","x":<int>,"y":<int>,"observed":"<what is at the marker center now>","reason":"<what is at the target's real position>"} — the target is NOT under the marker; x,y are the corrected full-screen coordinates of the intended target, read from the image.
- {"verdict":"fail","observed":"<reason>"} — the target is not visible in this image.
Do not confirm merely because the marker is close; if the target is outside the ring, adjust.`

const judge = async (b64, step, cx, cy, size) => {
  const res = await fetch(API + '/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: AbortSignal.timeout(120_000),
    body: JSON.stringify({
      model: MODEL,
      max_tokens: 400,
      temperature: 0,
      chat_template_kwargs: { enable_thinking: false },
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: verdictPrompt(step, cx, cy, size) },
            { type: 'image_url', image_url: { url: 'data:image/png;base64,' + b64 } },
          ],
        },
      ],
    }),
  })
  if (!res.ok) throw new Error('model call: HTTP ' + res.status + ' ' + (await res.text()).slice(0, 200))
  const d = await res.json()
  return d.choices?.[0]?.message?.content ?? ''
}

const parseVerdict = (text) => {
  for (const line of text.split('\n').reverse()) {
    const s = line.trim().replace(/^```json\s*|```$/g, '')
    if (!s.startsWith('{')) continue
    try {
      const v = JSON.parse(s)
      if (['confirmed', 'adjust', 'fail'].includes(v.verdict)) return v
    } catch {}
  }
  // fallback: any JSON object with a verdict field
  const m = text.match(/\{[^{}]*"verdict"[^{}]*\}/)
  if (m) { try { return JSON.parse(m[0]) } catch {} }
  return null
}

// ---------- the loop ----------
console.log(`coords_check_loop: target="${TARGET}" start=(${x},${y}) max=${MAX} vm=${VM} api=${API}`)
try {
  await mcp('initialize', { protocolVersion: '2025-03-26', capabilities: {}, clientInfo: { name: 'coords-check-loop', version: '1' } })
  for (let step = 1; step <= MAX; step++) {
    const { b64, size, file } = await captureMarked(x, y, step)
    console.log(`step ${step}: marker at (${x},${y}) on ${size} -> ${file}`)
    const answer = await judge(b64, step, x, y, size)
    const v = parseVerdict(answer)
    if (!v) throw new Error('no JSON verdict from model: ' + answer.slice(0, 200))
    console.log('OBSERVED: ' + (v.observed || '') + (v.reason ? ' | REASON: ' + v.reason : ''))
    console.log('ITERATIONS: ' + step)
    if (v.verdict === 'confirmed') {
      console.log('RESULT: VERIFIED')
      console.log('COORDS: ' + x + ' ' + y)
      process.exit(0)
    }
    if (v.verdict === 'fail') die(2, 'target not visible: ' + (v.observed || ''))
    // adjust
    const nx = Number(v.x), ny = Number(v.y)
    if (!Number.isInteger(nx) || !Number.isInteger(ny)) throw new Error('adjust without int coords: ' + JSON.stringify(v))
    if (Math.abs(nx - x) < 2 && Math.abs(ny - y) < 2) die(2, 'model insists on the same point but says not done')
    console.log(`step ${step}: adjust (${x},${y}) -> (${nx},${ny})`)
    x = nx
    y = ny
  }
  die(2, 'exhausted ' + MAX + ' corrections without confirmation')
} catch (e) {
  die(3, 'error: ' + (e?.message || String(e)).slice(0, 300))
}
