from main import build_indic_indic_rag_fuzzy, _INDIC_VECTOR_STORES

LANG_PAIRS = [
    ("hi", "or"),   # Hindi - Odia
    ("hi", "ta"),   # Hindi - Tamil
    ("hi", "bn"),   # Hindi - Bengali
    ("hi", "ml"),   # Hindi - Malayalam
    ("hi", "te"),   # Hindi - Telugu
    ("hi", "gu"),   # Hindi - Gujarati
    ("hi", "as"),   # Hindi - Assamese
    ("hi", "kn"),   # Hindi - Kannada
    ("hi", "mr"),   # Hindi - Marathi
    ("hi", "pa"),   # Hindi - Punjabi
    ("ta", "hi"),   # Tamil - Hindi
    ("bn", "or"),   # Bengali - Odia
    ("ta", "ml"),   # Tamil - Malayalam
    ("ta", "pa"),   # Tamil - Punjabi
]

print("Initializing Indic-Indic RAG databases")

for src, tgt in LANG_PAIRS:
    print(f"\n Building fuzzy RAG for {src} → {tgt} …")
    try:
        vs, df = build_indic_indic_rag_fuzzy(
            src_indic=src,
            tgt_indic=tgt,
            sample_per_lang=4500,
            sim_threshold=0.78
        )
        print(f"Built {len(df)} aligned pairs for {src}→{tgt}")
    except Exception as e:
        print(f"Failed to build {src}->{tgt}: {e}")
        continue

print("All vector stores are now saved to: ./chroma_indic_pairs_fuzzy/")
