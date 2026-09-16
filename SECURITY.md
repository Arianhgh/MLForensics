# Security policy

## Supported releases

MLForensics is currently an alpha project. Security fixes are applied to the
latest 0.1.x release and the current development branch. Older snapshots may not
receive patches. Upgrade to the newest release before reporting an issue that may
already have been fixed.

## Reporting a vulnerability

Please report vulnerabilities privately through the repository host's private
security-advisory feature. Include the affected version, operating system and
Python version, a minimal reproduction, the security impact, and any suggested
mitigation. Do not attach real credentials, personal data, proprietary datasets,
or an unredacted production capsule.

If no private advisory channel is available, contact the maintainers through a
private channel listed on the package or repository page and ask for a secure
reporting route. Please do not open a public issue until a fix or disclosure plan
has been agreed.

## Security boundaries

MLForensics treats capsules as evidence, not as authorization or proof of origin.
SHA-256 verification detects accidental or malicious modification after capsule
creation; it does not authenticate the creator. Store and transmit capsules with
the access controls appropriate for their contents.

Several features intentionally execute code:

- `run`, replay commands, shrink predicates, and bisect evaluators execute the
  supplied command;
- Python parity specifications import the supplied module and resolve an object;
- optional framework runtimes parse ONNX or TorchScript model files;
- remote artifact stores and integration clients can send evidence to external
  services.

Use only trusted commands, modules, plugins, capsules, and model files. Run
untrusted evaluation material inside a disposable, least-privileged environment
with network and filesystem access restricted. MLForensics is not a sandbox.

## Sensitive evidence

Before sharing or uploading a capsule, inspect it for:

- explicitly opted-in environment variables;
- sampled CSV, JSON, or JSONL rows and inferred schemas;
- stdout, stderr, exception messages, and trace data;
- configuration and metadata values;
- replay inputs, model/application state, and embedded artifacts;
- local paths, Git changes, host details, and dependency inventory.

Secret-name redaction and environment allowlisting reduce accidental exposure but
cannot recognize every credential or sensitive value. Prefer short-lived test
credentials, data minimization, encrypted storage, and retention limits.

## Dependency and release hygiene

Install releases from a trusted index, verify the expected package/version, and
keep optional ML runtimes current. Maintainers should build in a clean
environment, inspect the wheel and source distribution, test the installed wheel,
and avoid publishing local capsules, caches, signing material, or credentials.
