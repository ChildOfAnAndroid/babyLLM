# Historical Diagnostics

These 15 files preserve the original experiments byte-for-byte; SHA256SUMS
records their hashes. They target older model APIs or repeat model setup.
They are historical references, not supported runnable commands.

Do not run fix_optimizer_and_restart.sh: it kills training and deletes an
optimizer checkpoint without verifying the stated backup. migrate_add_expansion.py
also targets the old TANGLING checkpoint schema.

Use tools/profile_model.py for explicit weight/buffer inspection and synthetic
profiling, and tools/diagnose_model_stability.py for checkpoint stability checks.
The separate latent-probe views remain under utils/. Their shared model setup,
cache reading, labels and vendor loading now have shared owners.
