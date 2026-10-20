"""
Reference implementation: store, write (writer + checker + programmatic fallback), maintenance
(invalidation on key change and re-derivation), read (BM25 over s_k + fixed template + hash
assertion), JSON mismatch arm, .mem export/import, cost counting. The four (D, B) cells are
instantiated from the same batch of s_k.
"""
