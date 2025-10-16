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

## Docstring instructions

- When writing docstrings, use the Google style format and use Sphinx-compatible style. The napoleon extension is used. Use cross-references where appropriate, e.g. `:class:`ClassName`` or `:func:`function_name``.
- Keep the documentation focused, clear, and relevant to the function or class being described.
- Use proper grammar and punctuation in docstrings.

### Example of Proper Documentation

```python
def calculate_area(circle: Circle) -> float:
    """
    Calculate the area of a :class:`Circle` instance.

    Args:
        circle: The circle instance to calculate the area for.

    Returns:
       The area of the circle, calculated as π * radius^2.
    """
    import math
    return math.pi * radius ** 2
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
