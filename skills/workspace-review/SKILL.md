---
name: workspace-review
description: Inspect small workspace text files and summarize their goals and constraints.
---

# Workspace review

1. Identify the files relevant to the user's question. Use list_files if needed.
2. Request independent read_file calls together so the executor can run them concurrently.
3. Summarize the goals, explicit constraints, and open questions using the file contents.
4. Cite the workspace-relative filenames in the answer.
5. Report read failures and missing evidence honestly. Do not infer success from a request.

This skill is advisory. It grants no additional tools, write access, or execution permissions.
