# Release validation

- Date: 2026-10-08.
- 230 Python files parsed using `ast.parse`; 0 failures.
- 80 original project/preprocessing Python and shell scripts are byte-identical to the prepared local copies.
- Patient-level data, cohort identities, predictions, logs and internal review documents excluded.
- Four public run-settings references retain hyperparameters/hashes but omit cohort identities and run signatures.
- Three current/historical conda YAML copies omit installation-prefix fields; original local records remain unchanged.
- Three redistributed third-party snapshots retain their license files; MoLE source excluded pending explicit licensing, wrapper retained.
- MRI commands and training were not executed: private inputs, Linux MRI tools and model environments are not part of this publication check.
- Original sources retain server paths; portability is documented, not silently claimed.
- No project-wide open-source license selected.
