## Dataset Selection Report

**Selected Use Case:** Domain-Specific Instruction Following with Structured Outputs
**Task:** Given a natural-language instruction, produce a JSON object matching a predefined schema.

**Candidate Datasets Considered:**

1. **databricks/databricks-dolly-15k** (15,000 samples)
   - License: CC0 (public domain)
   - Format: instruction/input/output triplets
   - Domain: General purpose (domains like job search, travel, shopping)
   - Pros: Licensed freely, instruction-output format matches needs
   - Cons: General domain, not domain-specific; requires schema conversion for structured outputs

2. **tatsu-lab/alpaca** (52,000 samples)
   - License: Non-commercial research only (52k subset of ShareGPT)
   - Format: instruction/input/output triplets
   - Domain: General purpose
   - Pros: Well-known, substantial size
   - Cons: Non-commercial license restricts some use; general domain

3. **public/output_processed** (various sizes)
   - License: Varies
   - Format: code/output pairs
   - Domain: Code generation
   - Pros: Code-focused
   - Cons: Not instruction-following format; code-specific schemas

4. **Custom/ synthesized dataset** (targeted size)
   - License: Project-internal
   - Format: instruction + JSON schema output
   - Domain: Domain-specific (selected: " technical support troubleshooting")
   - Pros: Full control over domain, schema, size; no licensing concerns
   - Cons: Requires dataset creation

**Selected Approach: Custom Structured Output Dataset**

**Rationale:**
- **Domain specificity:** Can focus on a well-defined domain (technical support troubleshooting for consumer electronics)
- **Schema-based outputs:** JSON schema validation provides objective evaluation (exact match)
- **License control:** No third-party licensing restrictions
- **Size control:** Can create exactly the number of examples feasible for CPU training (500-1,000 examples)
- **Reproducibility:** Generated from documented rules, can be regenerated

**Dataset Specifications:**
- **Domain:** Consumer electronics troubleshooting (common issues: connectivity, performance, setup)
- **Schema:** `{"category": str, "action": str, "parameters": dict, "estimated_time_minutes": int}`
- **Target size:** 800 examples (600 train, 100 validation, 100 test)
- **Format:** JSONL with fields: `instruction`, `expected_output` (JSON string)
- **Examples per category:** Balanced across 4-5 troubleshooting categories

**Data Fields:**
- `instruction`: Natural-language description of the technical problem
- `expected_output`: JSON string matching the schema above
- Rationale: The `instruction` field maps to the model's input; `expected_output` is the ground truth for exact-match evaluation.

**Filtering Rules (to be applied during preprocessing):**
- Remove examples where instruction is >512 tokens (post-tokenization)
- Remove examples where expected_output is not valid JSON
- Remove duplicate instructions
- Normalize whitespace in instructions and outputs