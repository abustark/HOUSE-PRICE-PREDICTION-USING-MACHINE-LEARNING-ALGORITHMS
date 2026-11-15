# hpp-pyo3 — High-Performance Python C-Extension via PyO3

`hpp-pyo3` exposes the Rust validation engine, Indian unit parsing (`parse_sqft`), and embedded Linear/Ridge prediction math directly into Python as a compiled native C-extension module named `hpp_fast`.

## Performance Advantage
- **Zero Python Bytecode Overhead:** Regex execution, bounds checking, string trimming, and scalar dot products execute directly at native CPU speeds.
- **Thread-Safe & GIL-Free:** Can be invoked concurrently across multi-threaded Python web workers (e.g. Uvicorn, Gunicorn) without Python GIL contention.

## Building and Installing

Install `maturin`:
```bash
pip install maturin
```

Build and install into the current active Python virtual environment:
```bash
cd crates/hpp-pyo3
maturin develop --release
```

Or build distribution wheels:
```bash
maturin build --release
```

## Python Usage Example

```python
import hpp_fast

# 1. Native string and unit parsing
sqft = hpp_fast.parse_sqft("2100 - 2500")  # 2300.0
sqft_acres = hpp_fast.parse_sqft("2Acres") # 87120.0

# 2. Native fast prediction
price_lakh = hpp_fast.predict_fast(
    total_sqft=1200.0,
    bath=2,
    balcony=1,
    bhk=2,
    area_type="Super built-up Area",
    completion_year=0,
    location="Whitefield",
    society="",
    model="linear"
)

print(f"Predicted price: {price_lakh:.2f} Lakhs")
```
