# R54 external author-checkpoint replay availability audit

Date: 26 September 2026.

Status:

`R54_NOT_EXECUTABLE_NO_EXTERNAL_ARTIFACT`

## Required artifact

R54 was defined as a direct replay of an author-provided POPGym recurrent
checkpoint or equivalent immutable model artifact from the published paper
pipeline.

A valid R54 input would require at least:

- author-provided model weights/checkpoint;
- architecture/config identity;
- environment identity;
- enough serialization metadata to execute the checkpoint;
- immutable public provenance.

## Audit

Official repository checked:

`proroklab/popgym`

Current repository tree contains no:
- `.pt`;
- `.pth`;
- `.ckpt`;
- serialized model/archive artifact suitable for replay.

Official GitHub releases checked:
- `v1.0.7`;
- `paper` / `v0.0.2 (Paper)`;
- `v0.0.1`.

None exposes model/checkpoint release assets.

The repository includes the official training code and recurrent model
implementations, which were already used by R45 to train new official-code
checkpoints. That is not an author-checkpoint replay.

## Scientific decision

R54 is not failed. It is **not executable** under its declared scientific
question because the required external artifact is unavailable.

Do not substitute:
- another same-program checkpoint;
- an R45 reproduced checkpoint;
- a newly trained checkpoint;
- a guessed serialization.

If an immutable author checkpoint becomes publicly available later, R54 may be
reopened prospectively.

## Boundaries

The absence of a replay artifact does not weaken R45, which remains a bounded
official-code reproduced-checkpoint comparison.

It also does not establish POLAR superiority.
