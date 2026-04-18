# Analyst inputs

This directory is where human-entered fundamental research enters the
quantamental pipeline. **Git is the audit trail.** Every file here is
committed, attributable to its author via `git blame`, and versioned. The
loader (`kalshi_edge.fundamental.manual.loader`) reads YAML files from the
per-category subdirectories and converts each entry into a structured
`FundamentalInput` that the integration engine consumes exactly the same way
it consumes automated FRED pulls.

## Layout

```
analyst_inputs/
    README.md                       # this file
    economics/
        cpi_YYYY_MM.yaml            # one or more per release cycle
    elections/                      # (future)
    ...
```

The directory name is the category — it must match the `category:` field
inside each YAML file.

## File format

Top-level fields:

| Field      | Required | Meaning                                                    |
|------------|----------|------------------------------------------------------------|
| `category` | yes      | Kalshi category; must match parent directory name.         |
| `analyst`  | no       | Free-form string; surfaced in provenance as `analyst:<x>`. |
| `inputs`   | yes      | List of input entries (see below).                         |

Each entry under `inputs:`:

| Field              | Required | Meaning                                                                    |
|--------------------|----------|----------------------------------------------------------------------------|
| `name`             | yes      | Must be a declared `InputSpec` name for the category.                      |
| `value`            | yes      | Numeric anomaly value, in the spec's declared units.                       |
| `uncertainty`      | no       | 1σ estimate, same units as `value`; non-negative. Omit if unknown.          |
| `mechanism`        | no       | Defaults to the spec's mechanism. If set, must match the spec.             |
| `observation_at`   | yes      | ISO-8601 timestamp of the underlying observation (usually UTC with `Z`).   |
| `expires_at`       | cond.    | Absolute UTC timestamp after which this input is stale.                    |
| `expires_in_days`  | cond.    | Relative alternative to `expires_at`; must be > 0.                         |
| `scope`            | no       | Mapping of scoping tags, e.g. `{transform: mom}`.                          |
| `notes`            | no       | Free-form string; saved to provenance for audit only.                      |

Exactly one of `expires_at` / `expires_in_days` must be present.

## Example

See [`economics/example_cpi_manual.yaml`](economics/example_cpi_manual.yaml)
for a well-formed file that documents each field inline. Delete / rename the
example when you add a real input — the loader will still validate it as
correct, but keeping example data in a live pipeline is a footgun.

## Discipline

- **No free-form names.** `name` must be a spec declared in
  `kalshi_edge.fundamental.schemas.<category>`. If you need a new input,
  add the spec first, then the calibration, then the YAML entry.
- **Units must match the spec.** The loading calibration consumed the spec's
  units; entering a different unit silently breaks the integration math.
- **Uncertainty is honest.** If you can't estimate σ, leave it blank. The
  engine will treat the value as known; do *not* invent a number.
- **Expiry is honest too.** If you're entering a signal that's good until
  the next release, set `expires_at` to that release time. Don't pick a
  date so far in the future that the input can go stale unnoticed.
