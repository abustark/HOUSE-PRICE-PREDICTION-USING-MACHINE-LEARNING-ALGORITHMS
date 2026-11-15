# hpp-wasm — Client-Side WebAssembly Inference Engine

`hpp-wasm` compiles the core validation and linear/ridge prediction logic from `hpp-core` into a compact WebAssembly (`wasm32-unknown-unknown`) module.

## Features
- **0ms Round-Trip Time:** Run mathematical model inference directly inside the browser's JavaScript V8/SpiderMonkey runtime without HTTP requests.
- **100% Offline Capability:** Operates with zero network connectivity once cached.
- **Embedded Production Weights:** Linear Regression and Ridge Regression coefficients and standard scaler parameters are compiled directly into the binary.
- **Domain Validation:** Instant feedback on BHK, area bounds, unit conversions (Acres, Guntha, Sq. Yards, Perch, Sq. Meters), and layout sanity checks.

## Building WebAssembly

Install `wasm-pack` if not already installed:
```bash
cargo install wasm-pack
```

Build the release web bundle:
```bash
cd crates/hpp-wasm
wasm-pack build --target web --out-dir ../../static/wasm
```

This outputs:
- `static/wasm/hpp_wasm.js` (JavaScript glue code)
- `static/wasm/hpp_wasm_bg.wasm` (Compiled Wasm binary)

## Usage in JavaScript (`static/app.js`)

```javascript
import init, { wasm_predict, wasm_validate, wasm_parse_sqft } from '/static/wasm/hpp_wasm.js';

await init();

// Fast client-side prediction
const result = JSON.parse(wasm_predict(JSON.stringify({
  area_type: "Super built-up Area",
  location: "Whitefield",
  society: "",
  bhk: 2,
  total_sqft: 1200,
  bath: 2,
  balcony: 1,
  completion_year: 0,
  model: "linear"
})));

console.log(`Predicted Price: ${result.prediction.price_lakh} Lakhs`);
```
