# Failure Cluster Analysis

## Bottom 10 Questions

| # | Question (truncated) | Type | F | AR | CP | CR | Avg | Cluster |
|---|---|---|---|---|---|---|---|---|
| 1 | What responsibilities do organizations have under the law re | multi_context | 0.18 | 0.27 | 0.01 | 0.03 | 0.12 | C2 |
| 2 | What are the responsibilities of organizations regarding the | multi_context | 0.19 | 0.29 | 0.00 | 0.01 | 0.12 | C2 |
| 3 | What are the implications of cross-border data transfer on n | reasoning | 0.00 | 0.55 | 0.00 | 0.00 | 0.14 | C1 |
| 4 | What are the implications of cross-border data transfer on n | reasoning | 0.00 | 0.55 | 0.00 | 0.00 | 0.14 | C1 |
| 5 | What are the definitions and implications of personal data a | multi_context | 0.25 | 0.30 | 0.01 | 0.03 | 0.15 | C2 |
| 6 | What measures are implemented to protect personal data durin | reasoning | 0.00 | 0.60 | 0.00 | 0.00 | 0.15 | C1 |
| 7 | What are the difficulties and risks associated with the cros | reasoning | 0.00 | 0.60 | 0.00 | 0.00 | 0.15 | C1 |
| 8 | What are the specific compliance assessment methods and the  | reasoning | 0.00 | 0.62 | 0.00 | 0.00 | 0.16 | C1 |
| 9 | What are the difficulties and risks associated with cross-bo | reasoning | 0.00 | 0.65 | 0.00 | 0.00 | 0.16 | C1 |
| 10 | What are the risks associated with the transfer of personal  | reasoning | 0.00 | 0.65 | 0.00 | 0.00 | 0.16 | C1 |

## Clusters Identified

### Cluster C1: Reasoning compression failures

**Pattern:** Questions needing synthesis lose detail because the answer generator truncates the ground truth to a shorter first-sentence summary.
**Examples:**
- What are the implications of cross-border data transfer on national security, and how do data protection services address these concerns?
- What are the implications of cross-border data transfer on national security, particularly in relation to the measures for data protection services?
- What measures are implemented to protect personal data during cross-border data transfers, and how is compliance with data protection regulations evaluated?
**Proposed fix:**
- Increase answer synthesis depth for `reasoning` questions.
- Retrieve 2-3 supporting contexts instead of 1 for reasoning prompts.

### Cluster C2: Context precision drift on multi-context items

**Pattern:** Multi-context questions carry broader passages, so noisy tokens reduce precision even when the answer remains directionally correct.
**Examples:**
- What responsibilities do organizations have under the law regarding the management and protection of personal data in cyberspace, particularly in relation to national security?
- What are the responsibilities of organizations regarding the protection of personal data in the context of cybersecurity regulations, and how do these responsibilities relate to the general provisions outlined in the cybersecurity law?
- What are the definitions and implications of personal data as outlined in the regulations, and how do they relate to the assessment of data protection measures for sensitive personal data?
**Proposed fix:**
- Add a reranker or metadata filter before passing multiple contexts to generation.
- Reduce chunk size or compress contexts before final answering.
