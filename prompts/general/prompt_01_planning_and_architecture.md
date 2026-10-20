# Prompt Record 01: Project Planning & Chapter 6 Architecture

- **Prompt ID**: `P-GEN-01`
- **Date & Time**: 2026-10-20 20:12:02 IST
- **Track**: General / Project Planning (Tracks A–J)
- **AI Tool**: Antigravity (Powered by Gemini 3.8 Flash)
- **Commit Reference**: `59db77d`

---

### Purpose
Establish the complete Phase 1 project plan for Optimization Techniques Lab Assignment 8 ("Optimize Anything — The 14-Day Vibe-Coding Challenge"), covering all 10 compulsory tracks (A through J), mandatory Chapter 6 methods, beginner-friendly folder structure, technical requirements for Windows/WSL, Track E case study proposal, and strict vibe-coding documentation protocols.

---

### Original Prompt
```text
I need to complete my Optimization Techniques Lab Assignment 8: "Optimize Anything — The 14-Day Vibe-Coding Challenge."

Read the complete assignment requirements I provide and help me complete all 10 compulsory tracks, A to J.

My requirements:
1. Track A: Continuous optimization algorithms.
2. Track B: Discrete optimization, including Greedy, ILP, Branch-and-Bound, Dynamic Programming, and QUBO/Ising.
3. Track C: Quantum optimization, including QAOA, VQE, quantum neural networks, and the required experiments.
4. Track D: Submodular, parallel, distributed, FedAvg, MGDA, and GradNorm experiments.
5. Track E: A suitable real-world optimization case study.
6. Track F: Mini-compiler and compiler optimizations.
7. Track G: xv6 operating-system optimization.
8. Track H: DBMS optimization.
9. Track I: Quantum algorithms and quantum machine learning comparisons.
10. Track J: Software-engineering optimization.

First, DO NOT start coding immediately.

Phase 1 — Planning:
- Break the assignment into manageable tasks.
- Identify all required algorithms, experiments, metrics, tests, and deliverables.
- Propose a beginner-friendly project structure for the GitHub repository.
- Recommend suitable Python libraries and tools for each track.
- Identify operating-system requirements and dependencies, particularly for xv6, QEMU, compiler tools, databases, and quantum simulators.
- Propose a realistic 14-day implementation schedule.
- Suggest a manageable Track E problem that is different from Tracks F–J.
- Identify requirements that need clarification from my instructor.
- Create a checklist mapping every assignment requirement to its implementation, test, and report section.

Phase 2 — Implementation:
After presenting the plan, work on one track at a time.
- Generate working code, tests, and reproducible experiments.
- Establish a baseline and compare it fairly with the optimized approach.
- Record objective functions, decision variables, constraints, assumptions, runtime, and solution quality.
- Generate result tables and graphs from actual experiment outputs.
- Verify correctness independently and document failures and fixes.
- Do not fabricate results, claim unmeasured improvements, or mark incomplete experiments as completed.
- Use small, focused experiments wherever possible without omitting mandatory requirements.
- Keep the implementation understandable enough for me to explain during my viva.

Phase 3 — AI documentation:
- Create an AI tools register.
- Create a prompts/ directory with separate folders for Tracks A–J.
- Save original prompts, revised prompts, relevant AI responses, verification notes, and links to AI-assisted commits.
- Maintain a failure log containing at least three genuine AI-generated errors and their fixes.
- Plan a comparison of at least two AI tools on the same task.
- Clearly distinguish AI-generated changes from manual corrections.

Phase 4 — Final submission:
- Prepare a detailed README with installation and execution instructions.
- Prepare the Chapter 6 coverage checklist.
- Prepare a technical report covering all ten tracks.
- Include actual results, tables, graphs, correctness tests, limitations, and conclusions.
- Include instructions for reproducing every experiment.
- Prepare a short demonstration plan.

Important rules:
- Inspect my existing project files before modifying anything.
- Ask me questions when essential information is missing instead of inventing it.
- Never overwrite or delete existing work without checking with me.
- Do not assume that a library-based demonstration satisfies a requirement unless the assignment permits it.
- Do not skip any compulsory algorithm or experiment.
- Clearly distinguish simulator results, theoretical complexity, and actual hardware measurements.
- Run tests after every meaningful change.
- Give me simple, step-by-step instructions suitable for Windows, VS Code, and PowerShell.

Start with Phase 1 only. Give me the project plan, proposed folder structure, tool requirements, and checklist. Wait for my approval before making substantial changes.
### Mandatory Prompt Logging and PDF Documentation

Throughout this entire assignment, maintain a complete record of my AI-assisted development process.

1. Record every prompt I give you, including the initial planning prompt, follow-up prompts, debugging prompts, testing prompts, and revised prompts.
2. Record the date, track (A–J), purpose, relevant AI response or summary, files changed, and verification results for each prompt.
3. Save the records in an organized directory:

   prompts/
   - track_A/
   - track_B/
   - track_C/
   - track_D/
   - track_E/
   - track_F/
   - track_G/
   - track_H/
   - track_I/
   - track_J/
   - general/

4. Save each interaction in a readable Markdown or text file. Preserve the original prompt exactly as entered. Include the relevant AI response or excerpt and document any revisions.
5. Maintain a master file named `PROMPT_LOG.md` that indexes all recorded interactions.
6. Generate an updated PDF named `AI_Prompt_Log.pdf` containing the recorded prompts, relevant AI responses, dates, track names, code changes, and verification notes.
7. Update the Markdown records whenever we work on a new task. Regenerate the PDF at important milestones and before final submission.
8. Keep an AI Tools Register identifying the tool and model used, the dates, and the type of work performed.
9. Maintain a separate `AI_FAILURE_LOG.md` documenting at least three genuine AI-generated errors, their fixes, and the lessons learned. Never invent failures.
10. Clearly distinguish AI-generated code from manual corrections and record links to relevant Git commits when available.
11. Never include passwords, API keys, access tokens, or other secrets in the logs.

IMPORTANT:
- Do not fabricate missing prompts, AI responses, dates, or verification results.
- Do not claim to have recorded interactions that you cannot access.
- If you cannot automatically capture the complete conversation history, tell me which parts need to be copied or exported manually.
- Keep all logs organized and ready for inclusion in my public GitHub repository.

Start by reading the uploaded assignment PDF, reviewing the project requirements, and presenting the implementation plan. Do not begin substantial coding until I approve the plan.
```

---

### AI Response Summary
1. Examined assignment specification PDF thoroughly across pages 1 to 7.
2. Probed host Windows environment: verified Python 3.12, PyTorch 2.12, ReportLab, WSL2 Ubuntu, and Git availability.
3. Formulated a 10-track architectural breakdown and detailed experiment matrix.
4. Proposed a modular, beginner-friendly folder structure isolating all 10 tracks, tests, and prompts.
5. Outlined a 14-day schedule and proposed an EV Fleet Smart Charging and Route Scheduling problem for Track E to avoid overlap with Tracks F–J.
6. Defined the Chapter 6 coverage checklist and prompt logging workflow.
7. Requested clarification on Track E selection, dual-AI comparison tool, xv6 distribution, and GitHub repository state.

---

### Files Created / Modified
- Inspected workspace directory `c:\dev\ot-8` (empty).
- No code files modified prior to user approval.

---

### Verification Note
- Plan presented cleanly for human review.
- System environment verified: Windows 11 with WSL2 Ubuntu.

---

### Human Revisions / Instructions Received
User approved the plan with specific directives: proceed with EV Fleet for Track E, compare Antigravity vs Claude, use MIT RISC-V xv6 in WSL, initialize local repository first before publishing, and build/test the prompt-logging and PDF system before starting track implementations.
