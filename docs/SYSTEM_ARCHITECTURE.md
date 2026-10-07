# 12-6 system architecture contract

`src/twelve_six/system_architecture.py` is the executable Section 0 architecture boundary. It keeps the Base Model cognition core replaceable while keeping Post-Base learning, inference gateway, persistent cognition/memory, tools, Live Agent Plane, Evolution Plane, voice, operator UI and orchestration behind stable typed roles.

An unavailable role is represented explicitly as unavailable and is forbidden from advertising an implementation identity. `TwelveSixSystemAssembly.replace_cognitive_core()` replaces only the Base Model binding and exact core identities; all non-core slots must remain identical. This is the contract that lets later checkpoints or scales plug into the system without rewriting memory, tools, voice, UI or orchestration.
