// grounding_ab.mjs — behavioral coordinate test for the vision model.
// Draws red circles at known coordinates on a 1622x1018 image (win-b desktop),
// asks the model to report them, prints per-axis ratios.
//   ratio ~1.00 => model sees the image 1:1 (expected after patch 0015)
//   ratio ~0.78 => downscaled view (the old misdiagnosis)
//   systematic bias (e.g. y ~1.4x) => grounding error (the mrope bug, pre-0015)
// Run: node grounding_ab.mjs [baseURL] [size WxH]   (default http://192.168.2.208:19999/v1, 1622x1018)
// Requires: nothing beyond node (PNG encoded by hand; the model accepts any size).
import { writeFileSync } from 'node:fs'

const BASE = process.argv[2] || 'http://192.168.2.208:19999/v1'
const [W, H] = (process.argv[3] || '1622x1018').split('x').map(Number)
const MARKERS = [[1000, 400], [300, 700], [1400, 900]]

// --- minimal PNG writer (no deps): RGB, no interlace, zlib via node:zlib ---
import { deflateSync } from 'node:zlib'
function crc32(buf) {
  let c, table = crc32.table || (crc32.table = (() => {
    const t = []
    for (let n = 0; n < 256; n++) { c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0 }
    return t
  })())
  c = 0xffffffff
  for (const b of buf) c = table[(c ^ b) & 0xff] ^ (c >>> 8)
  return (c ^ 0xffffffff) >>> 0
}
function chunk(type, data) {
  const len = Buffer.alloc(4); len.writeUInt32BE(data.length)
  const body = Buffer.concat([Buffer.from(type), data])
  const crc = Buffer.alloc(4); crc.writeUInt32BE(crc32(body))
  return Buffer.concat([len, body, crc])
}
function png(w, h, rgb) { // rgb: Uint8Array w*h*3
  const raw = Buffer.alloc(h * (1 + w * 3))
  for (let y = 0; y < h; y++) {
    raw[y * (1 + w * 3)] = 0
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 3, o = y * (1 + w * 3) + 1 + x * 3
      raw[o] = rgb[i]; raw[o + 1] = rgb[i + 1]; raw[o + 2] = rgb[i + 2]
    }
  }
  const ihdr = Buffer.alloc(13)
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4)
  ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr),
    chunk('IDAT', deflateSync(raw)),
    chunk('IEND', Buffer.alloc(0)),
  ])
}
function drawMarkers(w, h, markers) {
  const px = new Uint8Array(w * h * 3).fill(245)
  for (const [cx, cy] of markers) {
    for (let y = cy - 16; y <= cy + 16; y++) for (let x = cx - 16; x <= cx + 16; x++) {
      if (x < 0 || y < 0 || x >= w || y >= h || (x - cx) ** 2 + (y - cy) ** 2 > 16 * 16) continue
      const i = (y * w + x) * 3
      px[i] = 220; px[i + 1] = 30; px[i + 2] = 30
    }
  }
  return png(w, h, px)
}

const q = 'The image contains three red circles. Report the pixel coordinates of each circle\'s center ' +
  'as integers (x, y) with (0,0) at the top-left of the image and x the horizontal axis. ' +
  'Also state the total width and height of the image in pixels as you see them. ' +
  'Answer as JSON: {"image_size": [w, h], "centers": [[x1,y1],[x2,y2],[x3,y3]]}'

const img = drawMarkers(W, H, MARKERS)
writeFileSync('grounding_ab_input.png', img)
const b64 = img.toString('base64')

const res = await fetch(BASE + '/chat/completions', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'qwen-3.8-flash-next', max_tokens: 300, temperature: 0,
    chat_template_kwargs: { enable_thinking: false },
    messages: [{ role: 'user', content: [
      { type: 'text', text: q },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,' + b64 } }] }],
  }),
})
if (!res.ok) { console.error('HTTP', res.status, (await res.text()).slice(0, 300)); process.exit(1) }
const d = await res.json()
const ans = d.choices?.[0]?.message?.content ?? ''
console.log('MODEL ANSWER:', ans)
// text-only baseline for this exact prompt is ~104 tokens; 1:1 image => +1634
// vision tokens (1632x1024 grid / 1024). So the expected total is ~1738.
console.log('prompt_tokens:', d.usage?.prompt_tokens, '(expected ~1738 if the image passed 1:1; a 1.0M-cap resize would be ~1040)')
const m = ans.match(/\{[\s\S]*\}/)
if (!m) { console.log('no JSON in answer'); process.exit(1) }
try {
  const j = JSON.parse(m[0])
  console.log('claimed image size:', j.image_size, '(true:', W + 'x' + H + ')')
  console.log('ground truth:', JSON.stringify(MARKERS))
  ;(j.centers || []).forEach((c, i) => {
    const [gx, gy] = MARKERS[i]
    if (c?.length === 2) console.log(`  gt=(${gx},${gy}) got=(${c}) ratio=(${(c[0] / gx).toFixed(3)},${(c[1] / gy).toFixed(3)})`)
  })
} catch (e) { console.log('parse fail:', e.message) }
