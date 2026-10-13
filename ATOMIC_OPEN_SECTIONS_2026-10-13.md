# 12-6 AI — повний реєстр самостійних розділів у незавершених планах

**Дата:** 2026-10-13. **Перевірено фактичний Google Docs readback:** всі 5 незавершених планів; у них 228 самостійних розділів і жодного нумерованого підпункту. Позначені DONE розділи були успадковані тільки від раніше прийнятих DONE-батьків; не відкривати їх знову.

**Робочий контракт:** NА, NБ, NВ — окремі рівноправні розділи, НЕ підрозділи. Старе N в PR, CI, старих документах та тестах — aggregate, який можна вважати DONE лише після незалежного доказового закриття усіх NА/NБ/NВ і спільної інтеграції. Кожному новому розділу відповідає повний текст завдання в Google Docs. Зберегти scope, already-integrated files, work ownership і upstream dependencies. Не вигадувати training/model/paid evidence.

**Разом:** 228, успадковано DONE: 75, ще не DONE: 153.

## План 2: 54 самостійних розділів

[Актуальний Google Doc](https://docs.google.com/document/d/1Ii2jGRHCcn6y_VhzbjrbwVr-Cymai20buKNuxsCJYlI/edit)

| Розділ | Зміст | Стан, успадкований із Google Docs |
| --- | --- | --- |
| 1А | Створити canonical registry джерел із source identity, location, acquisition method, language/modality, update cadence і provenance. | Статус: DONE |
| 1Б | Розділити accepted/candidate/rejected sources та не дозволяти невідомому source потрапити в training corpus. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 1; не відкривати. |
| 1В | Забезпечити reproducible source inventory і diff між corpus revisions. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 1; не відкривати. |
| 2А | Для кожного source зафіксувати license/terms/permission/legal basis і allowed use. | Статус: DONE |
| 2Б | Unknown rights fail-closed; research-only/private material не змішується з release/training authority без окремого дозволу. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 2; не відкривати. |
| 2В | Зберігати provenance до record/member рівня там, де це потрібно для audit/removal. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 2; не відкривати. |
| 3А | Завантаження/імпорт має бути resumable, hash-verified і idempotent. | Статус: DONE |
| 3Б | Не створювати другий data framework: використовувати canonical lineage/storage contracts. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 3; не відкривати. |
| 3В | Зберігати raw snapshot identity окремо від normalized derivatives. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 3; не відкривати. |
| 4А | Детерміновано нормалізувати encoding, Unicode, record boundaries і metadata. | Статус: DONE |
| 4Б | Визначати language/modality із confidence/evidence; unknown не маскувати як supported language. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 4; не відкривати. |
| 4В | Версіонувати normalization policy та invalidation downstream artifacts після зміни. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 4; не відкривати. |
| 5А | Виявляти та обробляти credentials/secrets/PII за explicit policy до training materialization. | Статус: DONE |
| 5Б | Зберігати audit counts/reasons без витоку sensitive payload у logs. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 5; не відкривати. |
| 5В | Підтримати removal/rebuild path після виявлення забороненого матеріалу. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 5; не відкривати. |
| 6А | Виконати exact content/member/record dedup із canonical equivalence identity. | Статус: DONE |
| 6Б | Dedup має бути global у межах accepted cohort, restartable і reproducible. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 6; не відкривати. |
| 6В | Зберігати duplicate families/provenance, а не просто видаляти без сліду. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 6; не відкривати. |
| 7А | Виявляти near duplicates, mirrors, templates і derivative families через deterministic/versioned method. | Статус: DONE |
| 7Б | Family caps і removal decisions не повинні залежати від випадкового input order. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 7; не відкривати. |
| 7В | Перевірити false-positive/false-negative tradeoffs на versioned audit samples. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 7; не відкривати. |
| 8А | Створити окремі reserved eval/holdout sources до training packing. | Статус: DONE |
| 8Б | Перевіряти exact/near contamination між training та evaluation families. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 8; не відкривати. |
| 8В | Заборонити teacher/self-generated material, що містить eval answers, автоматично повертати у training pool. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 8; не відкривати. |
| 9А | Визначити reproducible mixture за language/domain/source/family/modality із caps і sampling weights. | Статус: DONE |
| 9Б | Зберігати effective token/record contributions і reasons для exclusions. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 9; не відкривати. |
| 9В | Mixture changes створюють нову dataset identity/version. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 9; не відкривати. |
| 10А | Розбивати за duplicate/mirror/semantic family clusters, а не випадковими окремими records. | Статус: DONE |
| 10Б | Заборонити family leakage між train/validation/test. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 10; не відкривати. |
| 10В | Зберігати exact split manifest і deterministic seed/policy. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 10; не відкривати. |
| 11А | Побудувати candidate tokenizer architecture, normalization, special-token policy і vocab target без залежності від training runtime. | Статус: DONE |
| 11Б | Порівняти candidates за compression, multilingual behavior, code/text boundaries і robustness. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 11; не відкривати. |
| 11В | Зберігати candidate manifests/eval outputs. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 11; не відкривати. |
| 12А | Fit виконувати тільки на authoritative accepted corpus snapshot. | Статус: DONE |
| 12Б | Freeze exact vocab/model/normalization/special IDs/implementation hash як immutable TokenizerManifest. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 12; не відкривати. |
| 12В | Додати compatibility tests і заборонити silent tokenizer mutation після freeze. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 12; не відкривати. |
| 13А | Реалізувати exact tokenization→packing→sharding з order, boundaries, masks, truncation policy і shard hashes. | Статус: DONE |
| 13Б | Readers мають бути resumable і незалежними від конкретного optimizer/runtime. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 13; не відкривати. |
| 13В | Зберігати mapping source→tokens→shards для provenance/debug. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 13; не відкривати. |
| 14А | Надати stable exposure/target IDs для training runtime, щоб retry/resume не міг непомітно повторити або пропустити target. | Статус: DONE |
| 14Б | Exposure schema/versioning прив’язаний до Migration Contract Baseline v1; цей план виробляє ordered exposure manifest, а не керує optimizer state. Якщо потрібне сумісне schema extension, воно версіонується без очікування terminal Плану 1. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 14; не відкривати. |
| 14В | Перевірити boundary cases packing/shard rollover/truncation. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 14; не відкривати. |
| 15А | Зберегти ВСІ чинні результати PR #3220: 3 source families, 5 source identities, 32K BPE та physical S10–S14 receipt. Повторно перевірити їх exact SHA/CI; не проводити повторну acquisition або переписування успішного коду. | Статус: DONE |
| 15Б | Завершити negative/fail-closed source rights/catalog, five-source, physical split/mixture, privacy і license checks на existing PR; коли щось не працює — виправити один доведений failure у тій самій гілці, не створювати новий PR. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 15; не відкривати. |
| 15В | Зафіксувати candidate-only immutable receipts, ліцензійні обмеження і точні підтверджені тести; НЕ визнавати ці candidate-права автоматичним S3–S9 corpus/training або release дозволом. Вивести blocker list для Розділу 16. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 15; не відкривати. |
| 16А | Використати напрацювання старого розділу 15; не дублювати джерела, BPE, S10–S14 чи PR #3220. Кожне придатне джерело проходить оригінальні source grant, copyright, attribution/release, privacy/PII та immutability checks до навчального використання. | Статус: OPEN / WAITING_INDEPENDENT_SOURCE_EVIDENCEOWNER SCALE ADDENDUM 2026-10-11 (OPEN S16): Забезпечити один |
| 16Б | Кваліфікувати фізичні UA та EN train/validation/test окремими незалежними сім'ями; довести language/family balance, cluster-safe dedup, contamination firewall, non-overlap holdouts, gates S3–S9. Виключити видуманий контент або self-issued rights. | Статус: OPEN / WAITING_INDEPENDENT_SOURCE_EVIDENCE |
| 16В | Негативні tests: forged/revoked license; split leakage; train/test duplicate; missing Ukrainian evaluation; out-of-bound language distribution; no manifest promotion on error. Tests -> CI -> exact SHA readback. | Статус: OPEN / WAITING_INDEPENDENT_SOURCE_EVIDENCE |
| 17А | Повторно використати існуючі 32,768-byte BPE, three-family S10/S12/S13/S14 probes та їхні успішні SHA-bound job receipts; не тренувати токенізатор заново. | Статус: OPEN / WAITING_SECTION16OWNER SCALE ADDENDUM 2026-10-11 (OPEN S17): Frozen tokenizer/shards/exposure m |
| 17Б | З уже допущеного S3–S9 corpus детерміновано матеріалізувати train/validation/test, токенізаторні артефакти, shard manifests, exposure ledger та точні content/Git/SHA-256 fingerprints. | Статус: OPEN / WAITING_SECTION16 |
| 17В | Відтворити у clean checkout та independent process, перевірити byte-identical outputs; negative/recovery suite для змінених snapshots, membership, materialization, packing, overlap, revoked source та stale lineage. | Статус: OPEN / WAITING_SECTION16 |
| 18А | Перевірити весь accepted Plan2 1–17 scope на exact candidate; виконати повні права/приватність/dedup/eval/decontam/tokenizer/shards/recovery/integration CI без прихованих PASS або fake rights. | Статус: OPEN / WAITING_SECTIONS15–17 |
| 18Б | Зафіксувати final dataset/tokenizer/shard release з immutable production checksums, attribution/license bundle, release sign-off, consumer API та Plan9 real binding; об'єктивно підтвердити compatibility із Plan3/4 без неправомірного reopen. | Статус: OPEN / WAITING_SECTIONS15–17 |
| 18В | Після integration і live readback тільки тут позначити План 2 terminal DONE; оновити GitHub registry, INDEX, Drive та map статусів. Якщо rights/UA/eval/Plan9 бракує — NOT DONE і конкретний blocker, не нескінченний новий PR. | Статус: OPEN / WAITING_SECTIONS15–17 |

## План 3: 36 самостійних розділів

[Актуальний Google Doc](https://docs.google.com/document/d/1PizqJqX8q6xTJyF2DutYeXRelgPf8qQl-yioK9_EXjA/edit)

| Розділ | Зміст | Стан, успадкований із Google Docs |
| --- | --- | --- |
| 1А | Завершити canonical decoder-only Transformer contract: embeddings, RoPE, attention/GQA, RMSNorm, SwiGLU/MLP, LM head, context, tying, parameter count. | Статус: DONE. РІШЕННЯ ВЛАСНИКА 2026-10-11 — НОВИЙ ЦІЛЬОВИЙ МАСШТАБ, лише для OPEN-робіт. Історичні 20M/200M, п |
| 1Б | Initialization має бути deterministic за manifest/seed, незалежним від training launcher і random-init для canonical Base; foreign pretrained/instruct/aligned weights не можуть бути ancestor canonical Base checkpoint. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 1; не відкривати. |
| 1В | Додати serialization/version compatibility та architecture validation. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 1; не відкривати. |
| 2А | Перевірити forward/backward, masking, labels, loss normalization, padding/packing boundaries і no-future leakage. | Статус: OPEN |
| 2Б | Додати reference math/gradient checks, dtype/precision tests і pathological inputs. | Статус: OPEN |
| 2В | Numerical contract однаковий для local/server/distributed adapters у межах tolerance. | Статус: OPEN |
| 3А | Визначити versioned optimizer/scheduler/batch/accumulation/precision/clipping/regularization/budget semantics. | Статус: OPEN |
| 3Б | Recipe не залежить від конкретного compute provider; provider може змінити execution, але не scientific meaning. | Статус: OPEN |
| 3В | Зберігати exact recipe hash у run/checkpoint manifests. | Статус: OPEN |
| 4А | Реалізувати optimizer/scheduler із deterministic step semantics і warmup/decay policy. | Статус: OPEN |
| 4Б | Виявляти NaN/Inf/overflow/divergence, bounded recovery і fail-closed corruption handling. | Статус: OPEN |
| 4В | Кваліфікувати mixed precision/gradient scaling там, де підтримується. | Статус: OPEN |
| 5А | Визначити PREPARED/RUNNING/PAUSED/STOPPED/FAILED/COMPLETED та control epochs. | Статус: OPEN |
| 5Б | Заборонити одночасні writers одного run identity або duplicate optimizer advancement. | Статус: OPEN |
| 5В | Pause/Stop/restart мають deterministic durable receipts. | Статус: OPEN |
| 6А | Checkpoint містить model/optimizer/scheduler/RNG/exposure cursor/config/manifests із hashes і parent generation. | Статус: OPEN |
| 6Б | Публікація create-only/atomic; partial/corrupt checkpoint не стає resumable authority. | Статус: OPEN |
| 6В | Retention/GC не видаляє потрібний rollback parent. | Статус: OPEN |
| 7А | Відновлення в новому process/machine має продовжувати exact next exposure та training state. | Статус: OPEN |
| 7Б | Порівняти uninterrupted vs stop/resume trajectories в deterministic режимі. | Статус: OPEN |
| 7В | Corrupt/missing dependency fail-closed із actionable diagnostics. | Статус: OPEN |
| 8А | Реалізувати runtime exposure accounting проти stable exposure-contract fixtures Migration Baseline v1 і атомарно commit next exposure разом із training state; production exposure manifest Плану 2 підставляється лише в Плані 9. | Статус: OPEN |
| 8Б | Retry/provider migration не може створити duplicate causal-loss contribution. | Статус: OPEN |
| 8В | Вести unique-loss ledger і audit counters. | Статус: OPEN |
| 9А | Пакетувати ModelSpec, recipe, tokenizer/shard contract refs, budget, checkpoint policy, commands та expected evidence в portable run packet; для qualification цього плану refs можуть бути fixture identities, а real tokenizer/shards підставляються в Плані 9. | Статус: OPEN |
| 9Б | Один packet запускається local/free/server adapters без зміни scientific semantics. | Статус: OPEN |
| 9В | Provider-specific settings зберігаються окремо від canonical run identity. | Статус: OPEN |
| 10А | Виміряти memory, throughput, tokens/sec, checkpoint I/O і expected run time на підтримуваних profiles. | Статус: OPENДОДАТОК 2026-10-11 — послідовний benchmark усіх семи SCALE profiles: 500M, 750M, 1B, 2B, 3B, 4B, 5 |
| 10Б | Визначити admission rules, щоб завідомо impossible run не стартував. | Статус: OPEN |
| 10В | Resource metrics не змінюють scientific recipe без нової version. | Статус: OPEN |
| 11А | Створити replaceable local/runner/server launcher interface з common lifecycle/status/artifact receipts. | Статус: OPEN |
| 11Б | Provider failure не змінює run identity; migration/resume відбувається через canonical checkpoint. | Статус: OPEN |
| 11В | Немає provider-specific hidden training code path; paid/cloud provider не активується автономно без explicit owner authorization або чинної budget policy. | Статус: OPEN |
| 12А | Пройти tiny/synthetic deterministic training, crash/restart, provider-switch, numerical і checkpoint campaigns. | Статус: OPEN |
| 12Б | Довести reproducibility/semantic parity на exact candidate. | Статус: OPEN |
| 12В | Зафіксувати terminal Training Runtime release, готовий спожити artifacts Плану 2 у Плані 9. | Статус: OPEN |

## План 8: 33 самостійних розділів

[Актуальний Google Doc](https://docs.google.com/document/d/10rhE3yzfgMaaapxwfpoMv1smWeXLzrBZnq22yEa87lY/edit)

| Розділ | Зміст | Стан, успадкований із Google Docs |
| --- | --- | --- |
| 1А | Зберігати machine-readable capabilities, journeys, dependencies, environments, test vectors і evidence targets. | Статус: DONE |
| 1Б | Capability readiness пов’язана з exact source/artifact evidence, а не текстовим статусом. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 1; не відкривати. |
| 1В | Registry розширюється разом із продуктом та використовується іншими планами як contract. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 1; не відкривати. |
| 2А | Exact candidate запускається на clean hosted/available runner і виконує applicable journeys. | Статус: DONE |
| 2Б | Evidence artifact зв’язує source SHA, environment, commands, results і hashes. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 2; не відкривати. |
| 2В | Promotion не спирається на synthetic merge ref або stale candidate. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 2; не відкривати. |
| 3А | Автоматично перетворювати failed evidence на bounded defect packet із repro, expected/actual, candidate identity і affected acceptance. | Статус: DONE |
| 3Б | Repair worker змінює лише доведений gating scope, додає regression і рефризить candidate. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 3; не відкривати. |
| 3В | Цикл завершується independent promotion/readback, не самозвітом воркера. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 3; не відкривати. |
| 4А | Створити вузький agent/protocol і keyboard/NVDA-friendly operator harness для виконання exact package/test packet на реальному PC/server/accelerator; це qualification infrastructure, а не final product acceptance. | Статус: DONE |
| 4Б | Receipt містить host/OS/driver/hardware/package hashes/scenario/results та повертається в canonical evidence surface. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 4; не відкривати. |
| 4В | Physical agent не отримує unrestricted code/control authority. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 4; не відкривати. |
| 5А | Доставляти exact candidate/test packet на host і повертати execution receipts/issues/artifacts; локальний operator path має давати доступні text status/verify/stop controls без миші. | Статус: OPENOWNER SCALE ADDENDUM 2026-10-11 (OPEN S5 only): Qualification/control packets мають приймати profi |
| 5Б | Failure автоматично стає адресованим reproducible defect, не просто логом. | Статус: OPEN |
| 5В | Bridge підтримує reconnect/idempotency і no duplicate test/effect. | Статус: OPEN |
| 6А | Описувати product journeys як versioned scenarios із prerequisites/actions/observations/expected evidence. | Статус: OPEN |
| 6Б | Harness може працювати проти fixtures окремих планів і пізніше exact integrated product. | Статус: OPEN |
| 6В | Coverage map показує, які capability chains ще не мають потрібного evidence level. | Статус: OPEN |
| 7А | Інжектувати process kill, network loss, disk/corrupt artifact, stale state, provider/node failure та interrupted publication. | Статус: OPEN |
| 7Б | Перевіряти no duplicate exposure/effects, durable recovery, diagnostics і rollback. | Статус: OPEN |
| 7В | Fault campaign framework не вимагає final product — використовує component fixtures. | Статус: OPEN |
| 8А | Q0 component/contract, Q1 SIL, Q2 HIL, Q3 supervised physical operator/NVDA, Q4 field/canary evidence. | Статус: OPENOWNER SCALE ADDENDUM 2026-10-11 (OPEN S8 only): Розділяти для кожного SCALE profile статуси CONFIG |
| 8Б | Кожна capability декларує потрібний evidence level; simulation не підміняє physical requirement. | Статус: OPEN |
| 8В | Promotion policy invalidates stale evidence після behavior-changing mutation. | Статус: OPEN |
| 9А | Стандартизувати defect packet, test receipt, candidate freeze, repair parent, retest і promotion records. | Статус: OPEN |
| 9Б | Deduplicate repeated same defect і зберігати causality across retries/workers. | Статус: OPEN |
| 9В | Evidence artifacts мають hashes/retention/provenance. | Статус: OPEN |
| 10А | Candidate freeze забороняє unrelated mutation до terminal gate. | Статус: OPEN |
| 10Б | Integration використовує expected-head/topology checks і не втрачає qualified tree. | Статус: OPEN |
| 10В | DONE тільки після post-merge exact readback. | Статус: OPEN |
| 11А | Пройти SIL→defect→repair→retest, physical agent simulation/proxy, chaos та evidence-lifecycle campaigns. | Статус: OPEN |
| 11Б | Підтвердити integration із repositories/artifacts без залежності від actual production champion. | Статус: OPEN |
| 11В | Зафіксувати terminal Qualification Fabric release. | Статус: OPEN |

## План 9: 63 самостійних розділів

[Актуальний Google Doc](https://docs.google.com/document/d/1OL7-hhg-5jC70mWOLbN-X_flGZ2A8GL9HEp24ZJClns/edit)

| Розділ | Зміст | Стан, успадкований із Google Docs |
| --- | --- | --- |
| 1А | Звести accepted dataset/tokenizer/shards, ModelSpec, recipe, checkpoint/resume, eval firewall і inference у один exact launch manifest. | Статус: OPENОБОВ'ЯЗКОВЕ РІШЕННЯ ВЛАСНИКА ВІД 2026-10-11 (OPEN S1): замінити майбутню *основну ціль кампанії* l |
| 1Б | Перевірити lawful data authority, hashes, budgets, environment і stop criteria. | Статус: OPEN |
| 1В | Ніякий mock/fixture artifact не допускається як production input. | Статус: OPEN |
| 2А | Запустити короткий законний learned-20M pilot з реальними data/tokenizer/model contracts. | Статус: OPENOWNER SCALE AMENDMENT 2026-10-11 (OPEN S2): короткий bounded optimizer pilot виконується для перев |
| 2Б | Перевірити loss/numerics/exposure/checkpoint/resume/resource profile та зупинити при anomaly. | Статус: OPEN |
| 2В | Pilot не оголошується champion. | Статус: OPEN |
| 3А | Запустити full campaign за frozen recipe/budget з durable checkpoints і monitoring. | Статус: OPENOWNER SCALE AMENDMENT 2026-10-11 (OPEN S3): terminal pretraining campaign у новому цільовому маршр |
| 3Б | Будь-яка recovery/provider migration зберігає scientific run identity. | Статус: OPEN |
| 3В | Завершення training не означає promotion. | Статус: OPEN |
| 4А | Пройти frozen holdout/eval harness, capability/quality/robustness/resource metrics без вигадування окремої alignment/refusal policy для clean Base. | Статус: OPENOWNER SCALE AMENDMENT 2026-10-11 (OPEN S4): Base champion/eval порівнюється з обраним власником SC |
| 4Б | Independent audit приймає або відхиляє candidate; self-issued PASS заборонено. | Статус: OPEN |
| 4В | Promotion receipt робить accepted model live Base champion із rollback. | Статус: OPEN |
| 5А | Підключити champion до canonical inference/export/local service і перевірити реальні базові interactions. | Статус: OPENOWNER SCALE AMENDMENT 2026-10-11 (OPEN S5): локальна перевірка реального продукту стосується owner |
| 5Б | Зберегти resource profile/known limits та reproducible package refs. | Статус: OPEN |
| 5В | Це model-level usable product proof, ще не final 12-6 whole product release. | Статус: OPEN |
| 6А | Запустити accepted instruction engine Плану 6 на champion/descendant із lawful verified data. | Статус: OPEN |
| 6Б | Порівняти проти Base на frozen eval + regression. | Статус: OPEN |
| 6В | Promote тільки при доказовому net improvement. | Статус: OPEN |
| 7А | Провести preference optimization на accepted data/protocol. | Статус: OPEN |
| 7Б | Перевірити helpfulness/task quality разом із regressions/overoptimization. | Статус: OPEN |
| 7В | Зберегти champion fallback. | Статус: OPEN |
| 8А | Застосувати verifiable RL/reasoning лише на supported domains із надійним reward/evaluator. | Статус: OPEN |
| 8Б | Виміряти generalization, reward gaming і retention. | Статус: OPEN |
| 8В | Promotion проходить champion lifecycle. | Статус: OPEN |
| 9А | Навчити/доадаптувати модель на verified tool trajectories. | Статус: OPEN |
| 9Б | Перевірити schema adherence, planning/tool correctness, recovery та no authority bypass. | Статус: OPEN |
| 9В | Порівняти з Base/instruction descendants. | Статус: OPEN |
| 10А | Запустити bounded cycles Teacher Council→verified data→training→eval→promotion/reject. | Статус: OPEN |
| 10Б | Додати replay/anti-forgetting/consolidation і stop criteria. | Статус: OPEN |
| 10В | Кожний цикл має independent dataset/model/eval identity. | Статус: OPEN |
| 11А | Після terminal learned-20M зібрати measured loss/quality/throughput/memory/checkpoint/cost evidence та підготувати ~200M feasibility packet за актуальною accelerated strategy. | Статус: OPENOWNER SCALE AMENDMENT 2026-10-11 (OPEN S11): legacy postmortem 20M/~200M не є gate для першого сер |
| 11Б | Визначити data budget, architecture/growth-or-fresh-init path, LOCAL_FREE/free-GPU/paid options, checkpoint transport, serving/eval capacity та explicit stop criteria. | Статус: OPEN |
| 11В | Зафіксувати go/no-go на ~200M і окремо вирішити, чи потрібен проміжний <=100M risk probe; проміжний probe не є автоматичною сходинкою. | Статус: OPEN |
| 12А | Якщо feasibility evidence показує конкретний unresolved risk, вибрати мінімальний 35M/50M/100M probe; якщо ні — зафіксувати NOT_REQUIRED і перейти до ~200M без повної проміжної кампанії. | Статус: OPENOWNER SCALE AMENDMENT 2026-10-11 (OPEN S12): <=100M є винятково optional short smoke/probe за наяв |
| 12Б | Активований probe проходить bounded scientific/eval/checkpoint/resource protocol і не стає production champion лише через більший parameter count. | Статус: OPEN |
| 12В | DONE допускає два чесні outcomes: PROBE_COMPLETED з evidence або NOT_REQUIRED з documented reason; обидва не блокують ~200M path. | Статус: OPEN |
| 13А | Запустити ~200M лише після позитивного feasibility/admission decision, sufficient data budget і потрібної compute authority. | Статус: OPENOWNER SCALE AMENDMENT 2026-10-11 (OPEN S13): перший серйозний product-scale candidate відтепер SCA |
| 13Б | Пройти training/checkpoint/resume/eval/serving protocol і порівняти з learned-20M та будь-яким активованим risk probe за quality/time/cost. | Статус: OPEN |
| 13В | Більший розмір не є автоматичним promotion criterion; candidate може бути rejected або stopped. | Статус: OPEN |
| 14А | Провести terminal frozen evaluation, serving/resource qualification і independent audit exact ~200M candidate. | Статус: OPENOWNER SCALE AMENDMENT 2026-10-11 (OPEN S14): terminal model/eval/champion decision для першого сер |
| 14Б | Promote only on measured net improvement and compatibility; Base-20M/other accepted predecessors лишаються rollback/reference lineage. | Статус: OPEN |
| 14В | Зафіксувати champion/reject decision, resource profile, limitations, rollback parent і next-scale recommendation. | Статус: OPEN |
| 15А | Активувати 300–500M лише якщо ~200M evidence показує, що проміжний bridge знижує risk перед ~1B; інакше зафіксувати NOT_REQUIRED. | Статус: OPENOWNER SCALE AMENDMENT 2026-10-11 (OPEN S15): попередній 300–500M optional bridge більше НЕ є optio |
| 15Б | Перевірити scaling efficiency, quality gain, reliability й economics. | Статус: OPEN |
| 15В | Promotion conditional on evidence. | Статус: OPEN |
| 16А | Спочатку пройти systems admission Плану 7; без нього запуск заборонений. | Статус: DONE — NO-GO (PR #3216; main 5be7a108b59a4d4cc6ee7e68b6453fd2b7eb78e9; 28/28 LOCAL_FREE tests PASS, co |
| 16Б | За наявності ресурсів провести bounded/complete campaign і terminal evaluation. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 16; не відкривати. |
| 16В | Якщо ресурси/економіка не виправдані, terminal NO-GO є валідним завершенням цього еволюційного етапу. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 16; не відкривати. |
| 17А | Запуск необов’язковий для базової готовності; активується лише за ресурсним і research рішенням. | Статус: DONE — NOT_ACTIVATED (3B/7B/13B не активовано; PR #3210, main 562b1c0c4a2962866b357bb2c98d86d0c2b54ead |
| 17Б | Кожний реально запущений scale проходить той самий champion lifecycle. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 17; не відкривати. |
| 17В | Непроведені scale sizes мають explicit NOT-ACTIVATED, а не false DONE training claim. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 17; не відкривати. |
| 18А | Працює як conditional evolution program після explicit activation/resource authority. | Статус: DONE — CONDITIONAL NOT_ACTIVATED (30B/70B/100B не активовано; PR #3211, main 794fa2459c86d5327753c965e |
| 18Б | Використовує validated distributed/serving/eval contracts без redesign. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 18; не відкривати. |
| 18В | Кожен active candidate має terminal outcome. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 18; не відкривати. |
| 19А | Розглядати sparse MoE/dense alternatives через measured efficiency й ресурси. | Статус: DONE — NOT_ACTIVATED / NO-GO (300B/1T/MoE; PR #3212 merged main a9648b7f7351b1c24d14a8aa624ef4630429ce |
| 19Б | Full-scale execution тільки після explicit owner/resource activation; proxy/system proofs належать Плану 7. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 19; не відкривати. |
| 19В | Product completion не залежить від фактичного досягнення 1T parameters. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 19; не відкривати. |
| 20А | Для кожного campaign вести compute/storage/API/time cost, expected gain, actual gain і capacity impact. | Статус: DONE — Campaign economics STOP/NO_GO (PR #3213 merged main d3a7db568b6e0072ea9dbbd3f673d75b75f78469; 5 |
| 20Б | Stop decision є нормальним terminal result, якщо marginal gain не виправдовує ресурси. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 20; не відкривати. |
| 20В | Ніяка автономна система не може перевищити owner-approved budget; general «продовжуй» не є дозволом на materially paid compute. | Статус: DONE — успадковано з раніше прийнятого батьківського розділу 20; не відкривати. |
| 21А | Зібрати accepted champions/descendants, rollback parents, eval cards і scale decisions. | Статус: WAITING_UPSTREAM — NOT DONE (реальний accepted learned Base champion, production tokenizer/checkpoint, |
| 21Б | Визначити current production champion і teachers/critics/rollback models. | Статус: WAITING_UPSTREAM |
| 21В | Передати exact champion package у План 10. | Статус: WAITING_UPSTREAM |

## План 10: 42 самостійних розділів

[Актуальний Google Doc](https://docs.google.com/document/d/1vDUREMQTmhdyQDzgThphHw9jysLdgiPO52rt_JATUFE/edit)

| Розділ | Зміст | Стан, успадкований із Google Docs |
| --- | --- | --- |
| 1А | Перевірити підключення production champion через stable ModelGateway до зовнішнього Nika/Live Agent runtime. 12-6 repository не дублює Nika Core identity/memory/voice/UI implementation; він доводить brain/API compatibility, task/evidence/tool semantics і replaceability. | Статус: OPEN |
| 1Б | Model upgrade не стирає autobiographical/task continuity; predecessor доступний як rollback/teacher/critic. | Статус: OPEN |
| 1В | Перевірити live agent journeys на exact champion identity. | Статус: OPEN |
| 2А | Інтегрувати вже реалізовані component health/logs/metrics, backup/restore і migration primitives в один release candidate; не створювати тут альтернативний trainer, memory store або observability stack. | Статус: OPEN |
| 2Б | Backup/restore охоплює configs, manifests, registries, model artifacts, task/memory state та checkpoints. | Статус: OPEN |
| 2В | Versioned migrations мають corruption/partial-restore/rollback tests. | Статус: OPEN |
| 3А | Зібрати exact release package через reusable packaging/signing/update primitives Плану 1: runtime, champion/tokenizer/config, licenses, SBOM, manifests, cards та verification utilities без dev/secrets. | Статус: OPEN |
| 3Б | Signed installer/archive/update channel перевіряє signatures/hashes і робить atomic migration/rollback. | Статус: OPEN |
| 3В | Clean install/update/uninstall/reinstall кваліфікуються на exact package. | Статус: OPEN |
| 4А | Підв’язати exact product candidate до keyboard/NVDA-friendly operator/harness Плану 8 та наявних CLI/text paths. Окремий bespoke ChatGPT-like GUI не є обов’язковим, якщо доступний operator path повністю покриває declared workflow. | Статус: OPEN |
| 4Б | Критичні states/errors/logs мають semantic text parity і selectable/copyable output. | Статус: OPEN |
| 4В | NVDA-specific physical verification не ставиться тут автоматично; вона окремо нижче. | Статус: OPEN |
| 5А | Застосувати scenario harness Плану 8 до exact integrated candidate: champion inference, agent, memory/tools, self-learning, recovery, update. | Статус: OPEN |
| 5Б | Компонентний PASS не замінює composed journey. | Статус: OPEN |
| 5В | Behavior-changing repair інвалідовує affected evidence. | Статус: OPEN |
| 6А | На integrated candidate провести process/network/storage/provider/device faults і довгі tasks. | Статус: OPEN |
| 6Б | Перевірити no duplicate exposure/effects, checkpoint/task restore, service recovery та diagnostics. | Статус: OPEN |
| 6В | Failure створює defect→repair→retest loop через План 8. | Статус: OPEN |
| 7А | На реальному supported PC/GPU/storage/network виконати exact packaged candidate. | Статус: OPEN |
| 7Б | Evidence містить host profile, OS, drivers, hardware, package/model hashes і scenario results. | Статус: OPEN |
| 7В | Emulated/hosted evidence не підміняє потрібний physical proof. | Статус: OPEN |
| 8А | Перевірити exact candidate на supported server/accelerator environments: deploy, inference, training/resume, artifacts, provider loss, failover. | Статус: OPENOWNER SCALE ADDENDUM 2026-10-11 (OPEN S8): Cloud/server qualification must accept seven separately |
| 8Б | Local/server results не переносяться без parity evidence. | Статус: OPEN |
| 8В | Paid compute використовується тільки за explicit resource authority. | Статус: OPEN |
| 9А | Пройти install→first launch→model load→dialog/task→tools/memory→stop/resume→error/recovery→update→rollback→reinstall без dev environment. | Статус: OPEN |
| 9Б | Фізична keyboard-only/NVDA перевірка виконується на тому самому package hash, що претендує на release. | Статус: OPEN |
| 9В | Фіксувати actual findings; defect повертає candidate в repair/requalification. | Статус: OPEN |
| 10А | Порівняти reference/SIL/Windows HIL/server/cloud/optimized backends за model identity, output tolerances, task semantics, persistence/recovery і allowed features. | Статус: OPEN |
| 10Б | Unsupported combinations явно вилучити із support matrix. | Статус: OPEN |
| 10В | Promotion тільки після потрібних Q0–Q4 evidence levels. | Статус: OPEN |
| 11А | Активувати для exact release privacy-safe field evidence intake через qualification/evidence contracts Плану 8; цей розділ перевіряє production loop, а не будує другий telemetry/evidence platform. | Статус: OPEN |
| 11Б | Production fix проходить regression→SIL→applicable HIL/physical→staged update; rollback доступний. | Статус: OPEN |
| 11В | Field defect стає permanent regression scenario. | Статус: OPEN |
| 12А | Підготувати user/admin/research/developer docs, install/use/troubleshooting/recovery/training/evaluation/scaling manuals. | Статус: OPENOWNER SCALE ADDENDUM 2026-10-11 (OPEN S12): User documentation/model cards повинні окремо відображ |
| 12Б | Stable champion/release має model card, system card, data card, capabilities/limitations, resource profile, provenance і rollback info. | Статус: OPEN |
| 12В | Документація відповідає exact release, а не майбутнім намірам. | Статус: OPEN |
| 13А | Вести cost model compute/storage/network/teacher APIs/evaluation/retraining/serving/artifact retention. | Статус: OPENOWNER SCALE ADDENDUM 2026-10-11 (OPEN S13): Прорахувати кожен з 7 сценаріїв окремо (500M→750M→1B→2 |
| 13Б | Визначити hardware/provider upgrade policy, incident/support workflow і reproducibility packet. | Статус: OPEN |
| 13В | Capacity plan пов’язаний із реальними measured profiles Планів 3,4,7,9. | Статус: OPEN |
| 14А | Final matrix доводить усі declared supported capabilities потрібними Q0→Q4 evidence рівнями. | Статус: OPEN |
| 14Б | Немає unresolved P0/P1 blocker, false DONE, stale candidate або unsupported hidden promise. | Статус: OPEN |
| 14В | Evolution Engine після release може продовжувати descendants у дозволених resource/policy envelopes; базовий product DONE не залежить від максимального parameter count. | Статус: OPEN |
