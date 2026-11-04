// grounding_conv.mjs — decisive convention test (prompt-image-grounding.md §4 exp.1).
// Same 3 red circles, but the prompt asks for the model's NATIVE convention:
// relative coordinates in [0,1000] of the image. If the answer lands on
// truth/real-size*1000, the 1.44x is a translation artifact of forcing
// absolute px (fix = client-side denormalization). If it lands on
// truth/1.44x-size*1000 (i.e. ~1/1.44 of correct), the bias is perceptual.
//   node grounding_conv.mjs [baseURL] [size WxH]   (default 1622x1018)
//   --blank  : no markers, ask only the image size (exp.3)
import { writeFileSync } from 'node:fs'
import { deflateSync } from 'node:zlib'

const args = process.argv.slice(2)
const BLANK = args.includes('--blank')
const pos = args.filter(a => !a.startsWith('--'))
const BASE = pos[0] || 'http://192.168.2.208:19999/v1'
const [W, H] = (pos[1] || '1622x1018').split('x').map(Number)
const MARKERS = [[1000, 400], [300, 700], [1400, 900]]

// --- minimal PNG writer (same as grounding_ab.mjs) ---
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
function png(w, h, rgb) {
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

const qAbs = 'The image contains three red circles. Report the pixel coordinates of each circle\'s center ' +
  'as integers (x, y) with (0,0) at the top-left of the image and x the horizontal axis. ' +
  'Also state the total width and height of the image in pixels as you see them. ' +
  'Answer as JSON: {"image_size": [w, h], "centers": [[x1,y1],[x2,y2],[x3,y3]]}'
const qRel = 'The image contains three red circles. Report the center of each circle as RELATIVE ' +
  'coordinates in the range [0, 1000] of the image: x_rel = x_pixel / width * 1000, ' +
  'y_rel = y_pixel / height * 1000, as integers. ' +
  'Also state the total width and height of the image in pixels as you see them. ' +
  'Answer as JSON: {"image_size": [w, h], "centers_rel": [[x1,y1],[x2,y2],[x3,y3]]}'
const qSize = 'State the total width and height of this image in pixels as you see them. ' +
  'Answer as JSON: {"image_size": [w, h]}'

const img = BLANK ? png(W, H, new Uint8Array(W * H * 3).fill(245)) : drawMarkers(W, H, MARKERS)
writeFileSync('grounding_conv_input.png', img)
const b64 = img.toString('base64')
const q = BLANK ? qSize : qRel

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
console.log('MODE:', BLANK ? 'blank/size-only' : 'relative-0-1000', '| size:', W + 'x' + H)
console.log('MODEL ANSWER:', ans)
console.log('prompt_tokens:', d.usage?.prompt_tokens)
const m = ans.match(/\{[\s\S]*\}/)
if (!m) { console.log('no JSON in answer'); process.exit(1) }
try {
  const j = JSON.parse(m[0])
  if (j.image_size) console.log('claimed image size:', j.image_size, '(true:', W + 'x' + H + ')')
  const cs = (j.centers_rel || j.centers || []).filter(c => Array.isArray(c) && c.length === 2)
  if (!cs.length) { console.log('(size-only answer)'); process.exit(0) }
  // The model's enumeration order is NOT stable (x-sorted at 1622x1018,
  // input order at 1920x1080) => match greedily by nearest native expectation.
  const native = MARKERS.map(([x, y]) => [x / W * 1000, y / H * 1000])
  console.log('expected NATIVE (truth/real size):', JSON.stringify(native.map(([x, y]) => [Math.round(x), Math.round(y)])))
  const used = new Set()
  let maxErr = 0
  cs.forEach((c) => {
    let best = -1, bestD = Infinity
    native.forEach((n, i) => {
      if (used.has(i)) return
      const d = ((c[0] - n[0]) / 10) ** 2 + ((c[1] - n[1]) / 10) ** 2
      if (d < bestD) { bestD = d; best = i }
    })
    if (best < 0) return
    used.add(best)
    const errX = Math.abs(c[0] - native[best][0]) / 10, errY = Math.abs(c[1] - native[best][1]) / 10
    maxErr = Math.max(maxErr, errX, errY)
    console.log(`  gt=(${MARKERS[best]}) native_exp=(${native[best].map(v => v.toFixed(0))}) got=(${c}) err_in_0-1000_units=(${errX.toFixed(1)},${errY.toFixed(1)}) err_px=(${(errX * W / 1000).toFixed(1)},${(errY * H / 1000).toFixed(1)})`)
  })
  console.log(`VERDICT: ${maxErr <= 10 ? 'PASS' : 'FAIL'} — max error ${maxErr.toFixed(1)} units of 1000 (${(maxErr * W / 1000).toFixed(1)} px). PASS means the native 0-1000 convention is exact and the 1.44x is purely a translation artifact of forcing absolute px.`)
} catch (e) { console.log('parse fail:', e.message) }
