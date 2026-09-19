# MLForensics `mlcap-1` protocol

An `.mlcap` capsule is either a directory or a deterministic ZIP archive. The
two representations contain the same logical members. File names use `/` as a
separator, are relative to the capsule root, and must not contain `..`, NUL,
backslashes, absolute paths, or symbolic links.

## Required members

`manifest.json` is the integrity index. It contains `schema_version: 1`,
`format: "mlcap-1"`, a producer record, a list of all other member names, and a
`sha256` map with one digest for every listed member. `capsule.json` is the
canonical run-capsule document and is validated by `mlcap-1.schema.json`.
`run.json` is retained as a compatibility view and may be absent in a canonical
capsule. A reader must accept a capsule containing `capsule.json` without
`run.json`, and a legacy capsule containing only `run.json` may be migrated to
the canonical envelope.

Structured evidence uses stable role paths such as `code.json`,
`environment.json`, `data/fingerprints.json`, `training/events.jsonl`,
`failure/exception.json`, `replay/state.json`, and `artifacts/<sha256>`. Artifact
members are immutable content-addressed bytes; their digest and declared size
are checked against the `ArtifactRef` record.

## Compatibility and feature negotiation

The numeric schema version is the version of the canonical record and manifest,
not the producer package version. Readers reject versions newer than the
current version. Version-0 envelopes and records with omitted type/version
fields are migrated when `allow_migrations=True`; migration names are retained
in `migration_history` so a consumer can audit the conversion. Readers may
disable migrations when a strict interchange boundary is required.

`features` is an extensible JSON object. Its `flags` array describes optional
content present in the capsule. `required_reader_capabilities` and
`minimum_reader_version` are advisory compatibility gates and must be honored
by an application before attempting a workflow that depends on them. Unknown
feature fields are preserved as metadata.

## Integrity, limits, and streaming

Readers verify every manifest member by default, including members that are not
retained in a metadata-only load. `include_artifacts=False` and
`include_files={...}` allow selective retention without giving up integrity
verification. Per-file and total expansion limits protect both directory and
ZIP readers; callers can lower them for untrusted input. ZIP encryption,
duplicate members, unsafe names, symlinks, and declared sizes beyond the limits
are rejected before payload materialization.

## Optional authenticity and provenance

The manifest may contain `provenance` metadata and a detached `signature`.
`RunCapsule.save(signer=...)` passes canonical unsigned manifest bytes to a
caller-owned signer. A signer can return bytes, text, or a mapping with
`algorithm`, `encoding`, and `value`. `RunCapsule.load(verifier=...)` passes the
same bytes and signature mapping to the caller-owned verifier. No cryptographic
library is required by the protocol or by local capsules; digest integrity and
authenticity are deliberately separate claims.
