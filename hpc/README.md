# HPC launchers

These files preserve the original batch-job configuration used on the Bocconi HPC environment.

Before rerunning them on another system, review:

- SLURM account, partition, and QoS names;
- requested CPU/GPU/memory/time resources;
- Conda/virtual-environment names;
- absolute `$HOME` paths;
- input/output filenames.

They are retained for reproducibility rather than as portable, zero-configuration launch scripts.
