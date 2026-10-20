# AI Failure & Correction Log
**Assignment 8 Requirement**: Document at least three genuine cases where AI-generated code was wrong, unsafe, inefficient, or misleading, along with the independent verification test that caught it, the fix applied, and the lesson learned.

---

## Failure Entry Template
```markdown
### Failure Record [01 | 02 | 03]: <Descriptive Title>
- **Track**: Track X (<Track Name>)
- **Date**: YYYY-MM-DD
- **AI Tool & Model**: <Tool Name> (<Model Version>)
- **Prompt Reference**: [Link to prompt log](file:///path/to/prompt)
- **The AI Mistake**: Describe the erroneous, unsafe, or hallucinated code.
- **Why It Failed / Subtle Bug**: Root cause analysis.
- **Verification Method (How It Was Caught)**: Unit test, compiler error, kernel panic, or analytical check.
- **The Fix**: Code diff or corrected logic.
- **Viva Lesson Learned**: What this teaches about relying on AI for optimization and systems code.
```

---

## Active Failure Records

*(This log will record at least three genuine failures encountered during implementation of Tracks A–J. Zero fabricated failures are permitted.)*

### Log Status
- [ ] **Failure 1**: Pending implementation of Tracks A–D (Tracking subtle bugs in gradient calculation, Hessian conditioning, or QUBO sign convention).
- [ ] **Failure 2**: Pending implementation of Tracks F–H (Tracking compiler pass invalidation or xv6 kernel race/panic).
- [ ] **Failure 3**: Pending implementation of Tracks C, I, or J (Tracking quantum parameter-shift gradient dimension mismatch or DBMS plan hallucination).
