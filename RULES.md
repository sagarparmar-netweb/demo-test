# SIMPL AUTOCODE - MASTER RULES
## READ THIS FILE FIRST EVERY SESSION

---

## ⚠️ CRITICAL: DO NOT MODIFY WITHOUT PERMISSION

This file defines the authoritative rules for the Simpl Autocode pipeline.
Any changes must be approved by the user and logged.

---

## 1. PROMPT FORMAT

**Location:** `config/PROMPT_TEMPLATE.txt`

### System Message (EXACT):
```
You are an expert radiology coder specializing in ICD-10 and CPT coding for radiology reports.
```

### User Message Structure (MUST follow exactly):
1. Start with: `Study these similar cases:`
2. Include 3 RAG examples with `---` separators
3. Then: `Code this report:`
4. Then: Full report text
5. Then: `SIMILAR CASES CPT CODES (most likely):`
6. Then: `Available ICD codes (from similar cases):`
7. Then: Rules section (exact wording from template)
8. Then: Output format examples

**DO NOT:**
- Change wording of any section headers
- Add extra instructions or context
- Remove or reorder sections
- Paraphrase the rules

---

## 2. RAG RETRIEVER

**Config:** `config/rag_config.yaml`

### Index Files:
- `indices/nyxmed_71k_bm25.pkl`
- `indices/nyxmed_71k_faiss.index`
- `indices/nyxmed_71k_metadata.pkl`

### Settings (LOCKED):
| Setting | Value |
|---------|-------|
| BM25 top-k | 15 |
| Rerank top-k | 5 |
| Num examples | 3 |
| Reranker model | `BAAI/bge-reranker-v2-m3` |
| Use reranker | true |

### Query Format:
```
INDICATION: {indication} | IMPRESSION: {impression} | LATERALITY: {laterality}
```

**DO NOT:**
- Change the index prefix without rebuilding indices
- Modify the query format
- Change the reranker model
- Disable the reranker in production

---

## 3. PREPROCESSOR RULES

**File:** `src/preprocessor.py`

### CPT Disambiguation Rules:

#### Mammography (Fix 1):
- Check PRIMARY exam only (text before comma/semicolon) for modality
- If LLM predicts tomo codes (77061/77062/77063), TRUST the LLM
- Only override if clear evidence of different modality

#### Combined Studies:
- Split exam description at comma/semicolon
- Classify primary modality only
- Don't let secondary study keywords affect primary classification

**DO NOT:**
- Add new disambiguation rules without testing
- Remove the "trust LLM tomo prediction" logic
- Change how primary exam is detected

---

## 4. POSTPROCESSOR RULES

**File:** `src/postprocessor.py`

### ICD Filtering Rules (in order):

1. **Filter External Cause Codes** - Remove W, X, Y codes
2. **Filter Incidental Findings** - Remove unless CPT-relevant:
   - Cardiomegaly (I51.7) - allow cardiac CPTs
   - Fatty Liver (K76.0) - allow abdominal CPTs
   - Aortic/Coronary Calcifications (I25.10) - allow cardiac CPTs
   - Thyroid Nodules (E04.1, E04.2) - allow 76536 (thyroid US)
   - Hardware (Z96.x) - allow unless just noted as finding
3. **Diagnosis Hierarchy** - Definitive diagnosis overrules symptom
4. **Consolidate Laterality** - Merge bilateral codes
5. **Remove Conflicts** - Handle mutually exclusive codes
6. **Prioritize Anatomic** - M/N codes before R codes

**DO NOT:**
- Remove any filtering step
- Change the order of operations
- Modify the incidental findings CPT allowlist without approval

---

## 5. LLM CONFIGURATION

**File:** `src/llm_predictor.py`

### HuggingFace Endpoint:
- Model: Llama 3.3 70B (fine-tuned)
- Backend: HuggingFace Inference Endpoints
- Token: Set via `HF_API_TOKEN` environment variable

### Generation Parameters:
| Parameter | Value |
|-----------|-------|
| max_new_tokens | 150 |
| temperature | 0.1 |
| top_p | 0.9 |
| do_sample | true |

**DO NOT:**
- Change generation parameters without testing
- Use a different model endpoint
- Increase temperature above 0.2

---

## 6. SESSION STARTUP CHECKLIST

Before making ANY changes:

- [ ] Read this RULES.md file completely
- [ ] Verify working directory is `VFinal4/`
- [ ] Check `config/rag_config.yaml` is unchanged
- [ ] Check `config/PROMPT_TEMPLATE.txt` is unchanged
- [ ] Run 5 test samples to verify pipeline works

---

## 7. COMMON MISTAKES TO AVOID

| Mistake | Correct Approach |
|---------|-----------------|
| Reading wrong VFinal directory | Always use `VFinal4/` |
| Modifying prompt wording | Use exact template from `PROMPT_TEMPLATE.txt` |
| Changing RAG query format | Keep as compact format |
| Adding extra prompt context | Model was trained on specific format |
| Removing postprocessor rules | All rules are intentional |

---

## 8. RUNPOD DEPLOYMENT

### Sync Command:
```bash
rsync -avz --delete ./VFinal4/ runpod:/workspace/VFinal3/
```

### Critical Files to Sync:
- `src/*.py` - All pipeline code
- `config/*.yaml` - Configuration files
- `indices/*.pkl` - RAG indices (large files)

---

## 9. VERSION HISTORY

| Date | Change | Approved |
|------|--------|----------|
| 2026-01-12 | Initial locked configuration | Yes |
| 2026-01-12 | Production run 3108 records completed | Yes |

---

## 10. CONTACT

If unsure about any rule, ASK THE USER before making changes.
