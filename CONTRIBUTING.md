# Contributing to LLMBenchCore

Thank you for your interest in contributing to LLMBenchCore!

## How to Contribute

### Reporting Issues

- Use the GitHub issue tracker to report bugs
- Include a minimal reproducible example when possible
- Describe your environment (Python version, OS, etc.)

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run any existing tests to ensure nothing is broken
5. Commit your changes (`git commit -am 'Add new feature'`)
6. Push to the branch (`git push origin feature/my-feature`)
7. Create a Pull Request

### Code Style

This project uses:
- **yapf** for code formatting (see `.style.yapf`)
- 2-space indentation
- 100 character line limit

Format your code before submitting:
```bash
yapf -i -r .
```

### Adding New AI Engines

When adding support for a new AI provider:

1. Create `AiEngineProviderName.py` following the existing engine patterns
2. Implement the engine class with:
   - `__init__` accepting model, reasoning, and tools parameters
   - `configAndSettingsHash` property for cache keys
   - `AIHook(prompt, structure)` method returning `(result, chainOfThought)`
   - Compatibility with optional model config fields (`prompt_prefix`, `experiment_tag`)
3. Add the engine to `run_model_config()` in `TestRunner.py`
4. Update `requirements.txt` with any new dependencies
5. Document the setup in the engine's docstring

### Writing Tests

Test files should:
- Include a `title` variable describing the test
- Define `prompt` (string) or `prepareSubpassPrompt(subpass)` (function)
- Define `structure` (JSON schema) or `None` for text output
- Implement `gradeAnswer(result, subPass, aiEngineName)` returning `(score, explanation)`

## Questions?

Open an issue for any questions about contributing.
