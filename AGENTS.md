# Project rules (MUST follow)
- SPEC.md and GEOM_SPEC.md are frozen contracts. Do not edit them.
- To change specs: bump spec_version / geom_spec_version and update fixtures in the same PR.
- Config must include spec_version and geom_spec_version, and mismatch must hard-fail.
- Determinism: no -0, NaN/Inf, scientific notation; never use built-in round(); never sort raw floats.
- Always run tests after edits.

## Test command in this environment
- pytest/venv/make are not available.
- Always run: ./test.sh
- Include the output in the response.
