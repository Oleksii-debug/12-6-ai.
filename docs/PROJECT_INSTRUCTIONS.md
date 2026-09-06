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

ДОВГОСТРОКОВА AGENT-FIRST ЦІЛЬ
12-6 має розвиватися не лише як raw мовна модель, а як замінне інтелектуальне ядро повноцінної agentic AI system. Canonical Base залишається чистим і не містить hard-coded browser/file/API поведінки. Агентність, інструменти, пам’ять, довготривалі task state, планування, verification, post-training і orchestration існують поверх immutable Base checkpoint у post-Base/runtime шарах. Мета — не «довше генерувати відповідь», а автономно виконувати складну роботу як керований цикл goal -> plan -> act -> observe -> verify -> critique -> revise -> continue/finish.

LONG-RUNNING TASKS
Складне завдання не повинно бути прив’язане до одного prompt/response або одного довгого context window. Агентний runtime має підтримувати persistent task state: goal, constraints, deadline, plan, completed steps, pending steps, sources, evidence, files, hypotheses, rejected hypotheses, current draft/result, open problems, tool history, verification results, resource usage і resumable checkpoint. Процес після restart має відновлюватися з durable state, а не починати роботу заново. Дозволяй довгі задачі, що тривають години або довше, але не реалізовуй це як безконтрольний while-true loop. Потрібні deadline/resource limits, retry policy, progress detection, STOP_NO_PROGRESS/REFRAME/ASK_USER/FAILED/COMPLETED стани.

AGENT ROLES І ПІДАГЕНТИ
На ранніх масштабах не тренуй окрему нейромережу для кожної ролі без evidence. Спочатку одна Base Model може виконувати кілька логічно незалежних ролей у різних контекстах: planner, researcher, writer, critic, reviewer, tool router, hypothesis proposer. Незалежність ролей забезпечуй окремим context/state і контрольованим evidence handoff; не передавай приватний scratch одного role іншому без потреби. Пізніше окремі specialized models дозволені лише якщо вимірювання покажуть перевагу над shared-weight role instances. Heterogeneous multi-model agent system — наступний етап, не стартова вимога.

TOOLS / РУКИ Й НОГИ
Agent layer має отримати typed, permissioned tools замість hard-coded сервісної логіки у weights. Плануй стабільні tool contracts для: web search/browser/fetch/download; filesystem; PDF/DOCX/spreadsheet/document parsing; OCR; audio/video metadata; speech-to-text/transcription; calculator/Python/code tests; databases/RAG; APIs; дозволеного computer/UI interaction; communication/connectors. Конкретний backend під tool contract може змінюватися без перенавчання Base. Не дозволяй unrestricted shell, випадковий executable з Інтернету або неконтрольований arbitrary code як універсальний інструмент. Нові tools проходять qualification і потрапляють у versioned Tool Registry.

COMPUTER І BROWSER CONTROL
Для browser/computer automation надавай перевагу semantic DOM/accessibility tree/UIA та стабільним element IDs/roles над координатами миші. Типові actions: open_page, search, read_page, click_element, type_text, select_option, download_file, upload_file, open_document, save_document, wait_for_element. Небезпечні або незворотні дії мають окремі permission/confirmation boundaries. Агент може бути корисним як web/file/operator agent навіть якщо його Base ще замала для сильного універсального reasoning.

VERIFICATION ПЕРЕВАЖАЄ CONFIDENCE
Не будуй self-check як «та сама модель сказала, що її відповідь хороша». Перевірка має ієрархію. Рівень 1: deterministic/external evidence — calculator, compiler, unit tests, schema validation, hashes, source lookup, citation matching, file/API status. Applicable deterministic FAIL не може бути перекритий heuristic/model confidence. Рівень 2: independent model review у новому context для якості, аргументації, структури, стилю та невизначених тверджень. Рівень 3: goal-level completion controller, який перевіряє, чи виконані всі acceptance criteria задачі. Поки hard gates не пройдені, task не має ставати COMPLETED.

HYPOTHESIS SEARCH І ДОСЛІДНИЦЬКИЙ РЕЖИМ
Для складних research/engineering задач агент повинен підтримувати кілька конкуруючих hypotheses, assumptions, evidence, contradictions, parent relations і score histories. Спочатку preferred hypothesis може бути неправильною; нове evidence повинно мати можливість її відкинути. Пошук літератури/джерел має бути iterative: query -> read -> extract references -> verify provenance -> branch/refine query -> compare sources -> synthesis. Фактичні та цитатні твердження перевіряй проти первинного джерела, коли це практично можливо. Final answer не повинен залежати лише від самопідтвердження моделі.

QUALITY LOOP ДЛЯ ДОВГИХ АРТЕФАКТІВ
Для проєктів, статей, звітів, заявок, коду та інших великих робіт використовуй staged production, а не one-shot generation: specification -> research -> source verification -> outline/design -> draft/implementation -> claim/test audit -> independent critique -> revision -> second verification -> formatting/integration -> final gate. Додатковий compute витрачається на реальну роботу, пошук, тести, alternative branches і виправлення, а не на штучне очікування заданої кількості хвилин. Completion визначається acceptance criteria, дедлайном і evidence, а не довжиною тексту чи часом «думання».

STYLE / AI-DETECTOR BOUNDARY
Не використовуй AI-detector score як остаточний доказ людськості, якості або авторства тексту. Для практичного quality control краще мати style auditor, який знаходить конкретні дефекти: шаблонні переходи, повтори, vague claims, неприродну симетрію, lexical monotony, generic conclusions, register mismatch, unsupported assertions. Детектор може бути лише слабким допоміжним сигналом, якщо взагалі використовується.

SMALL-MODEL AGENT VALUE
Не плутай parameter count із можливістю діяти. Навіть Base на десятках/сотнях мільйонів параметрів може бути корисним агентним компонентом для bounded tool routing, monitoring, file/document collection, metadata extraction, deterministic workflow execution, simple browsing, transcription orchestration, classification, structured form filling і repetitive operator tasks. Водночас не заявляй, що 20M/100M здатна самостійно виконувати складне наукове дослідження або писати world-class роботу без evidence. Зі зростанням Base від 20M -> 100M -> 500M -> 1B -> 3B -> 10B той самий agent runtime має отримувати сильніше інтелектуальне ядро без переписування рук/ніг.

ORCHESTRATION BOUNDARY
12-6 повинна мати чіткий інтерфейс до зовнішнього orchestration/runtime шару, але не змішуй цей репозиторій з окремими продуктами на кшталт Nika Core. 12-6 визначає model/post-Base contracts, tool protocol, task/evidence semantics, verification і agent-facing APIs. Зовнішній runtime може реалізовувати scheduler, queues, browser/computer adapters, connectors, permissions, persistent jobs та UI, використовуючи ці контракти. Модель має бути замінною: новий 12-6 checkpoint повинен підключатися до того самого orchestration contract після compatibility verification.

OFFLINE SELF-IMPROVEMENT
Пам’ять і task state можуть оновлюватися безперервно, але model weights не переписуй після кожної взаємодії. Verified successful work може стати candidate post-Base training data тільки через provenance -> independent critique -> deterministic verification where applicable -> accept/reject -> immutable versioned dataset -> offline training -> independent evaluation -> promote/reject/rollback. Teacher/self-generated output не є truth лише через походження від моделі. Canonical Base training eligibility для post-Base synthetic/agent traces лишається false без окремого explicit authority.

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
