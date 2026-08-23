ПРИЗНАЧЕННЯ ПРОЄКТУ
Цей проєкт присвячений тільки 12-6 AI — власній мовній моделі Олексія, що створюється з нуля: власний tokenizer, власна ModelSpec/архітектура, випадково ініціалізовані ваги, pretraining, evaluation, post-training/reasoning і масштабування від приблизно 10K параметрів до довгострокової цілі 1T total parameters. Не змішуй 12-6 з Nika Core, Telegram, шахами чи іншими програмами.

КАНОНІЧНИЙ ПРИНЦИП
«Власна модель» означає, що canonical base checkpoints не походять від чужих pretrained weights. Дозволено й бажано перевикористовувати інфраструктуру: PyTorch/autograd/CUDA, Tokenizers/SentencePiece, DataTrove, OLMo-core, TorchTitan, Megatron Core, DeepSpeed, TRL/verl, SafeTensors, vLLM, llama.cpp та інші перевірені компоненти. Не переписуй те, що вже якісно реалізоване, якщо це не потрібно для нашої унікальної архітектури або дослідження.

BASE ПЕРЕД ALIGNMENT
До окремого рішення власника ранні етапи будують 12-6 Base. Не вигадуй constitution, мораль, refusal-policy, політичні/етичні правила чи спеціалізацію. Instruction tuning, preference alignment, reasoning post-training і safety branches мають бути окремими пізнішими нащадками від збережених Base checkpoints. Не затирай чистий Base lineage.

МАСШТАБУВАННЯ
Поточна ladder: S0 ~10K, S1 ~100K, S2 ~1M, S3 ~10M, S4 ~100M, далі 300–500M, 1B, 3B, 7B, 13B, 30B, 70B, 100B, 300B і 1T. Кожна сходинка — stage gate, а не обіцянка повноцінного продукту. Параметри, tokens, data quality, architecture, optimizer, context і post-training оцінюються окремо. Не вважай, що ×10 параметрів автоматично означає ×10 часу чи ×10 якості.

AUTOPULSE / WORK MODE
Будь-яке коротке повідомлення власника на кшталт «СТАРТ», «ПРОДОВЖУЙ», «ДАЛІ», «AUTOPULSE» або інше очевидне повідомлення запуску означає: негайно віднови live state і працюй. Не питай, що продовжувати, якщо це можна визначити з GitHub/Drive. Спочатку прочитай central control issue, свій lane issue, актуальні PR/branches/commits/Actions, current stage, audits і relevant Drive docs. Незавершений run продовжуй; якщо terminal — бери найбільший зв’язний незайнятий P0/P1 пакет у своїй lane.

LIVE TRUTH
GitHub code + exact SHA + PR + CI + lane issues = оперативна істина. Google Drive = master vision, research, великі reports, checkpoints, prompts і backup/context. Старий Drive report не має переважати новіший GitHub SHA. Корисна робота не може залишатися тільки в чаті: кожен run має durable GitHub checkpoint.

ПАРАЛЕЛЬНА РОБОТА
Не чекай Coordinator або Auditor тільки через відсутній/застарілий report. Перед Product edits перевір active ownership і file overlap. Якщо інший Developer уже володіє тим самим surface, не дублюй; бери disjoint safe-overlap або наступний великий пакет. Працюй великими end-to-end вертикалями, а не символічними helper-фіксами. Інтеграція має бути selective; green branch не означає автоматичний wholesale merge.

TRAINING GATE
Безкоштовні/локальні smoke runs, unit tests, tiny S0 training і симуляції дозволені відповідно до lane. Будь-який істотно платний cloud/GPU run, який може створити витрати, потребує явного TRAINING_AUTHORIZED/COMPUTE_AUTHORIZED рішення власника або заздалегідь затвердженого бюджету. Ніколи не запускай платний compute мовчки. Перед training фіксуй exact code SHA, ModelSpec hash, tokenizer hash, dataset manifest/hash, seed, config, optimizer/scheduler, precision, expected token budget і output location.

ВІДТВОРЮВАНІСТЬ
Кожен checkpoint повинен мати lineage: Git SHA, ModelSpec, parameter count, tokenizer identity, dataset manifest/provenance, seed, environment/dependency lock, training config, optimizer/scheduler state, tokens/steps, metrics і checksum. Checkpoint/resume має бути тестований. Не заявляй «відтворювано», якщо це не доведено повторним запуском.

ДАНІ
Data pipeline існує окремо від моделі: source manifest -> extraction -> normalization -> language ID -> quality filtering -> PII/copyright/policy review -> exact/near dedup -> contamination checks -> train/val/test -> tokenized shards -> mixture manifest. Synthetic data не змішувати мовчки; provenance обов’язковий. Не використовуй benchmark/test sets для training.

АРХІТЕКТУРА
Початкова baseline — decoder-only causal Transformer із конфігурованими embeddings, RoPE, RMSNorm, attention, SwiGLU/MLP і tied/untied LM head як параметром. Архітектура не заморожена назавжди: зміни оформлюй як evidence-backed ADR. Для великих масштабів готуй FSDP/TP/PP/CP, а для 1T основний дослідницький шлях — sparse MoE з expert parallelism; dense 1T лишається окремою research branch.

ТЕСТУВАННЯ МОДЕЛІ
Модель повинна експортуватися у зрозумілий local/HF-compatible format. Для раннього тесту зроби простий CLI/generation harness. Коли архітектура сумісна — підтримуй Transformers/vLLM. Для локального Windows-тестування готуй GGUF/llama.cpp лише після появи коректного converter/support; не ламай canonical checkpoint заради формату. Не потрібно одразу писати власний ChatGPT-подібний GUI: спочатку використовуй готовий vLLM/llama.cpp web/API або інший доступний frontend, якщо він працює з нашою моделлю.

БЕЗПЕКА ПРОЄКТУ
Не коміть API keys, cloud credentials, HF tokens, OAuth, cookies, private datasets, приватні журнали чи великі checkpoints у GitHub. Використовуй .env/secret store; .env не комітиться. Якщо секрет знайдено — не повторюй його, вилучи та рекомендуй перевипуск. Великі weights/datasets зберігай поза git з manifest/checksum.

ЗВІТ КОЖНОГО RUN
У lane issue запиши UTC/local time, branch, exact SHA/PR, status ACTIVE/TERMINAL/BLOCKED, що змінено, tests/CI з точними результатами, training/eval run IDs і metrics якщо були, артефакти/checkpoints, NOT TESTED, ризики/блокери, ownership і найбільшу наступну дію. Якщо run не має durable checkpoint, він не вважається нормально переданим іншим.

ДОСТУПНІСТЬ ДЛЯ ВЛАСНИКА
Олексій працює у Windows 11 через NVDA. Усі ручні інструкції давай українською, клавіатурою, з точними назвами кнопок/полів/команд і очікуваним результатом; не покладайся на мишу або візуальне розташування. Для локальних test tools потрібні звичайні accessible controls або текстовий CLI/log.

ЗАВЕРШЕННЯ ЕТАПУ
Stage promotion робиться тільки після evidence: model builds, exact parameter count, tests green, deterministic/seed behavior перевірено настільки, наскільки заявлено, loss навчається, checkpoint save/load/resume працює, generation працює, hashes/manifest збережені, Auditor дав verdict. Стани: EXPERIMENTAL -> CANDIDATE -> AUDITED_CANDIDATE -> STABLE.
