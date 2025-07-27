LLM Collaboration: Rules of Engagement & Best Practices
Objective: This document defines the standard operating procedures and guiding principles for any Large Language Model (LLM) assisting with the development of the Nifty 200 Pullback Strategy project. The primary goal is to ensure that all enhancements are made in a structured, non-disruptive, and verifiable manner, preserving the integrity of the existing, validated codebase.

Rule 1: Adhere to the Master Context Prompt
Principle: The "Master Project Context Prompt" (project_implementation.md) is the single, immutable source of truth for this project.

Directive: All of your responses, code modifications, and suggestions must be fully consistent with the architecture, logic, and file structures defined within that document. Before providing any response, you must confirm that you have fully ingested and understood the context provided. Do not rely on generalized knowledge; rely only on the project context.

Rule 2: No Unsolicited Refactoring
Principle: The existing, functional code has been debugged through a long, iterative process and is considered stable.

Directive: You must not refactor existing code for purely stylistic reasons (e.g., to make it more "Pythonic," to change variable names, or to alter code structure) unless you are explicitly asked to do so. Your primary function is to enhance, not to restyle.

Rule 3: Implement Incremental and Additive Changes Only
Principle: The system's stability is paramount. Changes should be small, isolated, and easy to verify.

Directive: When modifying existing code, your changes must be additive unless you are explicitly instructed to perform a refactor.

DO: Add a new if condition, a new function, a new parameter to a function, or new logging statements.

DO NOT: Rewrite or restructure an entire function or script to accommodate a new feature.

Rule 4: Preserve the Core Strategy Logic
Principle: The core trading rules are the "alpha" of the system and have been carefully developed.

Directive: The core trading rules—the entry pattern, the two-leg exit strategy, and the position sizing formula—are to be considered immutable unless you are given an explicit instruction to change a specific part of a rule. You may not alter these rules as an incidental side effect of another code modification.

Rule 5: Explain and Confirm Before Implementing
Principle: Clear communication prevents wasted time and effort.

Directive: For any request that involves a significant code change, you must first explain your proposed solution in plain English and ask for confirmation before you generate the code. For example: "To add this feature, I will need to modify the run_backtest function to include a new check after the volume filter and add a new parameter to the config dictionary. Is this correct?"

Rule 6: Maintain Modularity and Separation of Concerns
Principle: The project's strength lies in its clear separation of concerns (data acquisition, indicator calculation, strategy simulation).

Directive: When adding new features, you must follow the existing modular design. For example, a new data requirement should be handled by a new, dedicated script or function, rather than being embedded directly into the backtester. A new analysis should be a separate script that consumes the output of the backtester.

By adhering to these rules, you will ensure that your contributions are valuable, safe, and aligned with the long-term goals and stability of the project.