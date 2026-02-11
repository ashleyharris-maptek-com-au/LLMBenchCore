# LLMBenchCore

An abstract base framework for benchmarking Large Language Models (LLMs) across multiple AI providers.

## Overview

LLMBenchCore provides the core infrastructure for creating domain-specific LLM benchmarks. It handles:

- **Multi-provider AI engine abstraction** - Unified interface for OpenAI, Anthropic Claude, Google Gemini, xAI Grok, and Amazon Bedrock
- **Test orchestration** - Parallel and sequential test execution with subpass support
- **Response caching** - Intelligent caching to avoid redundant API calls
- **Result reporting** - Automatic HTML report generation with graphs and detailed breakdowns
- **Structured output** - JSON schema validation for structured responses
- **Tool support** - Web search, code execution, and custom tools where available

This repository is designed to be consumed as a dependency by domain-specific benchmarks (e.g., spatial/geometry benchmarks).

## Supported AI Providers

| Provider | Engine File | Models |
|----------|-------------|--------|
| OpenAI | `AiEngineOpenAiChatGPT.py` | GPT-5 series |
| Anthropic | `AiEngineAnthropicClaude.py` | Claude Sonnet/Opus 4.5 |
| Google | `AiEngineGoogleGemini.py` | Gemini 2.5/3 series |
| xAI | `AiEngineXAIGrok.py` | Grok 2/4 series |
| Amazon Bedrock | `AiEngineAmazonBedrock.py` | Qwen, Llama, Mistral, Nova |

## Installation

```bash
pip install -r requirements.txt
```

### API Keys

Set the appropriate environment variables for the providers you want to use:

```bash
# OpenAI
export OPENAI_API_KEY=your_key_here

# Anthropic
export ANTHROPIC_API_KEY=your_key_here

# Google Gemini
export GEMINI_API_KEY=your_key_here

# xAI Grok
export XAI_API_KEY=your_key_here

# Amazon Bedrock (use AWS credentials)
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# More models
python your_benchmark.py --list-models

# Self hosted models
export LLAMACPP_BASE_URL=http://localhost:8080
export LLAMACPP_MODEL_NAME="deepseek-v2.5"
```

## Usage

### Creating a Custom Benchmark

Subclass `BenchmarkRunner` from `TestRunner.py`:

```python
from TestRunner import BenchmarkRunner, run_benchmark_main

class MyBenchmark(BenchmarkRunner):
    def get_benchmark_title(self) -> str:
        return "My Custom LLM Benchmark"
    
    def get_benchmark_subtitle(self) -> str:
        return "Testing domain-specific capabilities"
    
    # Optional: Add custom scoring
    def can_handle_custom_scoring(self, test_globals: dict) -> bool:
        return "my_custom_scorer" in test_globals
    
    def process_custom_scoring(self, index, subPass, result, test_globals, aiEngineName):
        # Custom scoring logic
        score = test_globals["my_custom_scorer"](result, subPass)
        return {"score": score, "scoreExplanation": "Custom scored"}

if __name__ == "__main__":
    runner = MyBenchmark()
    run_benchmark_main(runner, __file__)
```

### Writing Test Files

Create numbered test files (`1.py`, `2.py`, etc.) in your benchmark directory:

```python
title = "Basic Math Test"

prompt = """
What is 2 + 2? Respond with just the number.
"""

structure = {
    "type": "object",
    "properties": {
        "answer": {"type": "integer"}
    },
    "required": ["answer"]
}

def gradeAnswer(result, subPass, aiEngineName):
    if result.get("answer") == 4:
        return 1.0, "Correct!"
    return 0.0, f"Expected 4, got {result.get('answer')}"
```

### Running Benchmarks

```bash
# Run all tests on all available models
python your_benchmark.py

# Try to use batch pricing where available (but may take up to 48 hours.)
python your_benchmark.py --batch

# Run specific tests
python your_benchmark.py -t 1,2,3
python your_benchmark.py -t 5-10

# Run specific models
python your_benchmark.py -m gpt-5-nano
python your_benchmark.py -m "claude-*"

# Run in parallel 
python your_benchmark.py --parallel

# List available models
python your_benchmark.py --list-models

# Force bypass cache
python your_benchmark.py --force

# Offline mode (cache only)
python your_benchmark.py --offline

# And more options...
python your_benchmark.py --help

```

### Optional Model Config Fields

Model config dictionaries support these optional fields in addition to core keys such as
`name`, `engine`, `base_model`, `reasoning`, and `tools`:

- `max_output_tokens`: Per-request output token cap (engine support is provider-specific).
- `temperature`: Per-request sampling temperature (engine support is provider-specific).
- `prompt_prefix`: Prepended to each benchmark prompt in runner and batch flows.
- `experiment_tag`: Stored in run summary metadata for grouping/filtering experiment runs.

## Architecture

```
LLMBenchCore/
├── TestRunner.py           # Core benchmark runner framework
├── CacheLayer.py           # Response caching system
├── ContentViolationHandler.py  # Content policy violation handling
├── PromptImageTagging.py   # Image embedding in prompts
├── AiEngineOpenAiChatGPT.py    # OpenAI engine
├── AiEngineAnthropicClaude.py  # Anthropic engine
├── AiEngineGoogleGemini.py     # Google Gemini engine
├── AiEngineXAIGrok.py          # xAI Grok engine
├── AiEngineAmazonBedrock.py    # Amazon Bedrock engine
└── AiEnginePlacebo.py          # Placebo engine for baselines
```

## Features

### Reasoning Modes

Most engines support configurable reasoning effort (0-10 scale):

- **0/False**: Standard mode (fastest)
- **1-3**: Low reasoning
- **4-7**: Medium reasoning  
- **8-10**: High reasoning (most thorough)

### Tool Support

Enable built-in tools like web search and code execution:

```python
configs.append({
    "name": "gpt-5-nano-Tools",
    "engine": "openai",
    "base_model": "gpt-5-nano",
    "reasoning": 5,
    "tools": True,  # Enable all built-in tools
    "env_key": "OPENAI_API_KEY"
})
```

### Structured Output

Use JSON schemas for validated responses:

```python
structure = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["answer", "confidence"]
}
```

### Image Support

Embed images in prompts using the `[[image:path]]` syntax:

```python
prompt = """
Describe what you see in this image:
[[image:images/test.png]]
"""
```

## Placebo Engine

Use the Placebo engine with pre-defined responses to create configurable baselines:

```python
from AiEnginePlacebo import set_placebo_data_provider

def my_responses(model_name: str, question_num: int, subpass: int):
    responses = {
        ("baseline-positive", 1, 0): ({"answer": 4}, "Expected positive result"),
        ("baseline-negative", 1, 0): ({"answer": 5}, "Expected negative result"),
    }
    return responses.get((model_name, question_num, subpass), (None, ""))

set_placebo_data_provider(modelNames, my_responses)
```

Configure multiple placebo models by setting the modelNames variable
to a list of model names, or by adding configs manually with
`engine: "placebo"` in your runner overrides.

## Bill shock protections

The benchmark framework has protections to save your wallet, which need to be acknowledged in any research drawn from these results. The following compromises are made:

- Early fail:
  - Assuming failure of hard tasks if easy tasks are failed. This is call EarlyFail, and can be configured per test.
  - If you can't lay out 4 pipes in a square or 3 pipes in a triangle, you're not going to be able to lay out 400 pipes in the shape of a world map.
  - Early fail can be configured to sample multiple early runs, or have adjustable thresholds per test.
  - This only works when tests are configured that the early subpasses are easiest, so it's opt-in.
  - This can be turned off with --no-early-fail
- Propagation upwards:
  - Assuming a model can ace a test without tools or reasoning, assume that it will also pass with tools or reasoning.
  - If version 3 of the model can ace a test, assume that version 4 also can.
  - This works by sorting models based on capability, and propagating results between models with the same prefix.
  - Can be disabled with --no-propagate-upwards
- API instability lock out:
  - If the API fails 9 times in a row, assume it will fail again for the rest of the run.
  - This can stop a run wasting days retrying when you've hit a spend limit or the network goes down.
- Double caching:
  - Results are cached in your temp directory and in the git repo.
  - This allows runs performed on one machine to be used by another machine to save costs.
- Eternal caching:
  - Original design was for results to only be cached for a month, however the realities of API costs got in the way.
  - You can turn this off by disabling POOR_MODE in CacheLayer.py

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
