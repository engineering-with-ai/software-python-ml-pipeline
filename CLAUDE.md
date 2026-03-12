## Methodologies
### Implementation Methodology
When presented with a request YOU MUST:
1. Use context7 mcp server or websearch tool to get the latest related documentation. Understand the API deeply and all of its nuances and options
2. Use TDD Approach: Figure out how to validate that the task is complete and working as expected. Whether using a CLI tool like curl, or ssh command or writing unit/integration test
3. Start with the simplest happy path test. The test should fail on `unimplemented!()|raise NotImplemented| throw Error("Not Implmented")`. Scaffold out all functions (including full signature) with this body 
4. See the test fail with not implemented error.
5. Make the smallest change possible
6. Take time to think through the most optimal order of operations for implementation
7. Check if tests and `(npm |cargo cmd | uv run) checks` passes
8. Repeat step 5-6 until the test passes
9. You MUST NOT move on until tests pass

### Debugging Methodology

#### Phase I: Information Gathering

1. Understand the error
2. Read the relevant source code: try local `.venv`, `node_modules` or `$HOME/.cargo/registry/src/`
3. Look at any relevant github issues for the library

#### Phase II: Testing Hypothesis
4. Develop a hypothesis that resolves the root cause of the problem. Must only chase root cause possible solutions.  Think hard to decide if its root cause or NOT.
5. Add debug logs to determine hypothesis
6. If not successful, YOU MUST clean up any artifact or code attempts in this debug cycle. Then repeat steps 1-5

#### Phase III: Weigh Tradeoffs
7. If successful and fix is straightforward. Apply fix
8. If not straightforward, weigh the tradeoffs and provide a recommendation



## 🧱 Code Structure & Modularity
- **Follow SOLID Principles***
- **Never Break Up nested Values:** When working with a value that is part of a larger
  structure or has a parent object, always import or pass the entire parent structure
  as an argument. Never extract or isolate the nested value from its parent context.
- **Write Elegant Code** Write the most minimal code to get the job done
- **Get to root of the problem** Never write hacky workarounds. You are done when the tests pass 
- **Never create a file longer than 200 lines of code.** If a file approaches this limit, refactor by splitting it into modules or helper files.
- **Use cfg.yml file for config variable. You MUST NOT add config vars to env files.**
- **Use template-secrets.env file to keep track of the list of secrets:**
- **Use environment variables for secrets** Do NOT conflate secrets with config variables
- **Keep it generic class/type names: TimeseriesClient instead of TimeScaleClient**
- **Use Generics Judiciously:** Remember, while generics are powerful, they can also make code more complex if
  overused. Always consider readability and maintainability when deciding whether to
  use generics. If the use of generics doesn't provide a clear benefit in terms of
  code reuse, type safety, or API design, it might be better to use concrete types
  instead.

### 🧪 Testing & Reliability
- **Fail fast, fail early**: Detect errors as early as possible and halt execution. Rely on the runtime or system to handle the error and provide a stack trace.  You MUST NOT write random error handling for no good reason.
- **Unit Tests should be colocated in `src/`**
- **Integration Tests** should be located in `tests/`
- **Use AAA (Arrange, Act, Assert) pattern for tests**:
  - **Arrange**: Set up the necessary context and inputs.
  - **Act**: Execute the code under test.
  - **Assert**: Verify the outcome matches expectations.
- **Use testcontainers for integration tests** — spin up real databases/services in Docker, session-scoped for performance

## 💅 Style
- **Constants in code:** Write top level declarations in SCREAMING_SNAKE_CASE.

### 📚 Documentation & Explainability
- **Comment non-obvious code** and ensure everything is understandable to a mid-level developer.
- When writing complex logic, **add an inline `Reason:` comment** explaining the why, not just the what.
- **Write concise document comments for primarily for an LLM to consume, secondarily for a document generator to consume**



## Python  Anti Bias Guidelines 🐍

- **IGNORE Python's "duck typing" culture - use explicit type hints ALWAYS**
  - NOT: `def process(data):`
  - CORRECT: `def process(data: UnprocessedData) -> List[ProcessedItem]:`
- **No `Any`:**  `Any` typing annotation is NEVER ALLOWED. 
- **Most Python code omits return types - YOU MUST include them**
- **Override common practice: prefer pydantic/pandera models over dicts for structured data**
- **Training bias toward print() debugging - use proper logging instead**


### Python Testing Guidelines
- **Use actual/expected semantics**  `assert actual == expected`  or `assert_frame_equal(actual_df, expected_df)`

### Python Patterns

- **Prefer structural matching:** Use match/case statements (PEP 636)
- **Prefer validated types:** Use Pydantic or Pandera for type definitions
- **Prefer list comprehensions** for transforming list of objects
- **Use enums** to constrain sets of strings or numbers
- **Use `Final` typing annotation** on top level constants
- **Use Optional type** for parameters that can be None
- **Write concise Google Style Docstrings for an llm to consume:**

  ```python
  def calculate_stats(data: list[int]) -> Stats:
    """Calculates basic statistics for numeric data.

    Args:
        data: List of numbers to analyze

    Returns:
        Stats object with calculated metrics

    Raises:
        ValueError: If data is empty

    Example:
        >>> calculate_stats([1, 2, 3]).mean
        2.0
    """
  ```


## Python ML Pipeline Guidelines 🤖

### Data Processing & Validation
- **Use Pandera for data validation** with typed DataFrames: `DataFrame[ModelInputSchema]`
- **Use Polars for data processing** over Pandas for performance
- **Implement data transformation pipelines** with clear separation for testability
- **Handle missing data explicitly** - filter null values and validate schema

### Model Training & MLflow
- **Use MLflow for model tracking** and deployment
- **Implement champion/challenger pattern** - compare models and promote best performer
- **Version models automatically** in CI/CD pipeline
- **Use baseline model fallback** when performance degrades

### Data Pipeline Patterns
- **Use scheduled processing** with `schedule` library for daily/periodic runs
- **Define clean model interfaces** with well-defined Pandera types for inputs and outputs
- **Separate data processing from model training** for modularity

### Database & Time Series
- **Use TimescaleDB for time series data** with conflict resolution strategies
- **Implement upsert patterns** - `ON CONFLICT DO UPDATE` for data updates
- **Store processed data with timestamps** in UTC

### Testing Patterns
- **Test data transformations** with expected vs actual DataFrames
- **Use `polars.testing.assert_frame_equal`** for DataFrame comparisons


### Model Development
- **Use Jupyter notebooks for exploration** - `model_evaluation.ipynb`
- **Separate exploration from production code**
- **Implement model comparison logic** 
- **Use statistical validation** for model performance


