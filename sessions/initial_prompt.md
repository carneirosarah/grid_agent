Your task is to build an application with two panels: an editable data table and a chat interface. You should create a script to generate the dataset with least 300 rows and 8 columns. Use a product inventory, an expense ledger, a library catalog, or anything comparable. The user instructs an agent in natural language, and the agent edits the table.

Requirements:

- The agent handles multi-step commands. Example: "Add a Margin column computed from Price and Cost, flag every row where Margin is below 15 percent, then sort by Margin ascending."
- The agent expresses its edits as structured operations that your application code applies deterministically. The model should never regenerate the table contents directly.
- Pending changes appear as a preview the user can accept or reject. Applied changes can be undone.
- When an instruction is ambiguous, the agent asks a clarifying question rather than guessing.

Instructions:

- Implement the application using Python and Langgraph
- You should use Gemini 3 flash free API with structured output
- You should follow the code best practices, and prioritize code quality, simplicity and optimization
- Create a rich documentation, comment the code step by step 
- You should implement only two operations: update_where and sort. For example: "Increase eletronics prices by 10%, then sort descending"
- Split the implementation in the following steps. For each of the steps, bring the code and how to test it.
    1. Data generation
    2. Operations Implementations
    3. Schemas tests 
    4. Deterministc Operations Engine Implementation
    5. Semantic Validator
    6. State Management, preview and Undo
    7. Integration with Gemini API
    8. Langgraph Implementation
    9. End2End tests
    10. FastAPI implementation
    11. Frontend application
- You should persist the trace inside a .jsonl file
- You should suggest improvements both in the suggested approach and in the code.
- You should prioritize: tool and operation design, separation of planning from execution, state management, and how you handle incorrect model output delivered with confidence.
