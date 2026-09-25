# E7 metadata-panel canonical run selection

Date: 25 September 2026, America/Lima.

This selection is recorded before the metadata-panel result is inspected.

Canonical metadata-only panel-selection run:

- GitHub Actions run: `36192944203`
- event: `push`
- head SHA: `39f7616f9413381fede26cfb6acc36eebc36c0fc`
- workflow: `E7 IBL metadata-only panel freeze`

Only this run may define the E7 development/confirmatory panel unless it fails
for a documented infrastructure/schema reason before producing a valid panel.

This run is metadata-only. It must not load or report spike times, firing-rate
features, neural decoders, or E7 D/C/R endpoints.
