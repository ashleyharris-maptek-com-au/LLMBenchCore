import hashlib
import inspect
from typing import Any, Callable


def _hook_accepts_context(hook: Callable[..., Any]) -> bool:
  try:
    signature = inspect.signature(hook)
  except (TypeError, ValueError):
    return False

  params = list(signature.parameters.values())
  if any(param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD) for param in params):
    return True
  return len(params) >= 3


def _call_ai_hook(engine: Any, prompt: str, structure: dict | None, context: dict | None):
  hook = engine.AIHook
  if _hook_accepts_context(hook):
    return hook(prompt, structure, context)
  return hook(prompt, structure)


class AiEngineMultiplexer:

  def __init__(self, factories: list[Callable[..., Any]], model: str, reasoning=False, tools=False):
    self._factories = list(factories)
    self.model = model
    self.reasoning = reasoning
    self.tools = tools
    self._engines = None
    self._selected_engine = None
    factory_names = [
      getattr(factory, "__name__", factory.__class__.__name__) for factory in self._factories
    ]
    self.configAndSettingsHash = hashlib.sha256(
      (str(model) + "|" + str(reasoning) + "|" + str(tools) + "|" +
       "|".join(factory_names)).encode("utf-8")).hexdigest()

  def _build_engines(self) -> list[Any]:
    if self._engines is not None:
      return self._engines

    engines = []
    for factory in self._factories:
      try:
        engine = factory(self.model, self.reasoning, self.tools)
      except Exception:
        continue
      if engine is not None:
        engines.append(engine)

    self._engines = engines
    return engines

  def _is_available(self, engine: Any) -> bool:
    available = getattr(engine, "Available", None)
    if available is None:
      return True

    try:
      result = available()
    except TypeError:
      result = available
    except Exception:
      return False

    return result is True or result == True

  def Available(self):
    for engine in self._build_engines():
      if self._is_available(engine):
        backend_hash = getattr(engine, "configAndSettingsHash", None)
        if backend_hash:
          self.configAndSettingsHash = backend_hash
        return True
    return False

  def _select_engine(self) -> Any:
    if self._selected_engine is not None:
      return self._selected_engine

    for engine in self._build_engines():
      if self._is_available(engine):
        self._selected_engine = engine
        backend_hash = getattr(engine, "configAndSettingsHash", None)
        if backend_hash:
          self.configAndSettingsHash = backend_hash
        return engine

    raise RuntimeError(f"No available backend for model '{self.model}'")

  def AIHook(self, prompt: str, structure: dict | None, context: dict | None = None):
    engine = self._select_engine()
    try:
      return _call_ai_hook(engine, prompt, structure, context)
    except LookupError:
      self._engines.remove(engine)
      self._selected_engine = None
      print(f"MULTIPLEXER: Falling back from {engine.__class__.__name__} to next available backend")
      return self.AIHook(prompt, structure, context)


class AiEngineMultiplexerFactory:

  def __init__(self, factories: list[Callable[..., Any]]):
    self._factories = [factory for factory in factories if factory is not None]

  def create(self, model: str, reasoning=False, tools=False) -> AiEngineMultiplexer:
    return AiEngineMultiplexer(self._factories, model, reasoning, tools)
