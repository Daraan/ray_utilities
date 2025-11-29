---
description: 'Python coding conventions and guidelines'
applyTo: "**/*.py"
---

# Python Coding Conventions

## Python Instructions

- Write clear and concise comments for each function.
- Ensure functions have descriptive names and include type hints.
- Use the `typing` module for type annotations (e.g., `list[str]`, `dict[str, int]`). Do not use the uppercase versions like `List` or `Dict`.
- When working with `dicts` create a `TypedDict` if the structure is known.
- Break down complex functions into smaller, more manageable functions.

## General Instructions

- Always prioritize readability and clarity.
- For mathematical or algorithm-related code, include explanations of the approach used.
- Write code with good maintainability practices, including comments on why certain design decisions were made.
- Handle edge cases and write clear exception handling.
- For libraries or external dependencies, mention their usage and purpose in comments.
- Use consistent naming conventions and follow language-specific best practices.
- Write concise, efficient, and idiomatic code that is also easily understandable.
- Do not remove comments that are unrelated and in a different parts of the code that are unrelated to your change.
- When you refactor code assure that you keep the type hint information and comments intact.

# Review Instructions - apply when reviewing Python code changes or you are requested as a reviewer
- When you encounter a `cast("some_import", variable)`, it means that `some_import` is actually used as an import for type casting, despite it being a string. For reviews do not comment on such unused imports.
- do not comment on unused variables starting with an underscore `_` as they are intentionally unused but descriptive.
- For larger PRs with many changes be thorough in reviewing all files, be sure to review and address all files changed.

## Docstring instructions

- When writing docstrings, use the Google style format and use Sphinx-compatible style. The napoleon extension is used. Use cross-references where appropriate, e.g. `:class:`ClassName`` or `:func:`function_name``.
- Keep the documentation focused, clear, and relevant to the function or class being described.
- Use proper grammar and punctuation in docstrings.

### Example of Proper Documentation

```python
def calculate_area(circle: Circle) -> float:
    """
    Calculate the area of a :class:`Circle` instance.
    Requires :attr:`Circle.radius` to be defined.

    Args:
        circle: The circle instance to calculate the area for.

    Returns:
       The area of the circle, calculated as π * radius^2.
    """
    import math
    return math.pi * circle.radius ** 2
```

## Code Style and Formatting

- Follow the **PEP 8** style guide for Python.
- Maintain proper indentation (use 4 spaces for each level of indentation).
- Ensure lines do not exceed 120 characters.
- Place function and class docstrings immediately after the `def` or `class` keyword.
- Use blank lines to separate functions, classes, and code blocks where appropriate.

## Edge Cases and Testing

- Always include test cases for critical paths of the application.
- Account for common edge cases like empty inputs, invalid data types, and large datasets.
- Include comments for edge cases and the expected behavior in those cases.
- Write unit tests for functions and document them minimally with docstrings roughly explaining the test cases.
- Prioritize and focus on the actual use case first, then add edge cases as needed.
- Run the tests
- When creating a new shell, prompt the user to activate the correct virtual environment.
