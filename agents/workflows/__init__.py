"""Workflow package: domain-specific entry points that drive the supervisor.

Each workflow module exposes a high-level ``run_*`` function plus
its own helpers. The CLI scripts in ``scripts/`` are thin wrappers
around these functions.
"""
