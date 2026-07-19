# Engineering Notes

## 1. Model and Framework Selection
The system uses Google Gemini Flash Lite through the official Google GenAI Python SDK.
I chose this model because it provides a good balance between latency, cost, and structured output capabilities. The model’s responsibility is intentionally limited to translating natural language into a structured execution plan rather than executing business logic, making a lightweight model sufficient for this task.
The application intentionally separates planning from execution:
- The LLM proposes a structured plan.
- The application validates that plan against the live dataset.
- A deterministic execution engine applies the changes to a copy of the DataFrame.
- The user explicitly accepts or rejects the preview before any data is committed.
This separation ensures that the LLM is never the source of truth. Structured outputs prevent many syntax errors, but semantic correctness is enforced entirely by the application.
The planner is isolated behind a Planner interface, allowing the LLM provider to be replaced without affecting the rest of the system.
Other options considered:
- OpenAI GPT-4.1 / GPT-4o — Excellent structured output support, but higher cost than necessary for this use case.
- Groq (GPT-OSS) — Considered as an alternative provider due to its low latency and competitive pricing.
- Local models (Ollama / Qwen) — Rejected because they increase setup complexity and generally produce less reliable structured outputs.
For orchestration I used LangGraph. Although the workflow is relatively small, representing it as explicit graph nodes (context preparation, planning, validation, preview creation) keeps the execution flow modular, testable, and easy to extend.

## 2. Unit Economics

Each user command performs:
- one LLM request;
- one semantic validation pass;
- one deterministic DataFrame transformation;
- optional preview acceptance.
The dominant operational cost is the LLM request.
Only lightweight context is sent to the model:
- table schema;
- column types;
- a small sample of rows.
The complete dataset is never sent.
At approximately 100× usage, I would expect:
- LLM inference cost to dominate infrastructure cost;
- pandas execution to remain negligible for datasets of this size;
- latency to be primarily determined by model response time.
Potential optimizations include:
- caching schema descriptions;
- reducing prompt size further;
- supporting multiple LLM providers with automatic fallback;
- replacing pandas with Polars or DuckDB for significantly larger datasets.

## 3. Main Failure Modes

### 1. Structurally valid but semantically incorrect plans
Structured outputs eliminate many syntax errors but do not guarantee that a plan is correct.
Examples include:
- nonexistent columns;
- incompatible value types;
- unsupported operations;
- invalid filter values.
The semantic validator checks every plan against the live DataFrame before execution. Invalid plans are rejected and never reach the execution layer.

### 2. Ambiguous user requests
Some instructions cannot be executed safely because they omit required information.
Rather than guessing, the planner returns a clarification response, allowing the user to provide the missing information before any operation is performed.

### 3. State consistency
Applying changes immediately would make recovery difficult if validation failed or the user changed their mind.
Instead, every command generates a preview on a copy of the dataset.
Only explicit user confirmation updates the committed state. Rejecting the preview leaves the original dataset untouched, and undo operates on committed snapshots rather than attempting to reverse individual operations.

## 4. What I Cut for Time

To keep the implementation focused, I intentionally limited the supported operations to:
- update_where
- sort

This allowed me to invest more effort into correctness, validation, testing, and architecture instead of implementing many operations with limited robustness.

I also chose not to implement:

- provider failover across multiple LLMs;
- persistent storage beyond the scope of the assessment;
- security controls as OWASP Top 10
- advanced spreadsheet operations such as computed columns or aggregations.

With one additional week I would prioritize:

- implementing additional operational metrics (average input tokens, average output tokens, average cost per request, average latency);
- integrating Langfuse for observability, tracing, and evaluation dashboards;
- replacing the vanilla frontend with a component-based framework to support pagination, richer validation feedback, and a better user experience;
- security controls as OWASP Top 10;
- adding more spreadsheet operations (insert, delete, rename columns, computed columns, aggregation);
- implementing provider abstraction with configurable model selection and automatic fallback;
- adding persistent version history;
- improving scalability by replacing pandas with Polars or DuckDB for larger datasets.