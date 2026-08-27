"""Compatibility imports; use :mod:`rag_system.web_results` instead."""

from rag_system.web_results import RankedWebResult, canonicalize_url, rank_web_results


__all__ = ["RankedWebResult", "canonicalize_url", "rank_web_results"]
