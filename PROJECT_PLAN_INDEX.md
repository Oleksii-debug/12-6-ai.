# 12-6 AI — Canonical Multi-Plan Index

## Authority

The former monolithic 96-Section execution plan is **SUPERSEDED FOR WORK SELECTION**.
Its content remains historical/audit evidence only.

The project now uses ten numbered plans in Google Drive folder:
https://drive.google.com/drive/folders/1dltwOgSMBmc39c35bwPkxkB5AWaZbxdX

Plans 1–8 are independent engineering plans. There is no priority order among them.
Safe parallel ownership, Migration Contract Baseline v1 and conflict-key rules are defined in `MULTI_PLAN_PARALLELISM_CONTRACT.md`.
Plan 9 is a training/champion convergence plan.
Plan 10 is the final whole-product integration/release plan.

## Plans

1. Перший план — Архітектура, контракти та supply chain
https://docs.google.com/document/d/1yrxH1SpC7ch6Q0hI5LhXC_01kpxOV9MqHwDSCaO0Cf0/edit

2. Другий план — Дані та токенізація
https://docs.google.com/document/d/1Ii2jGRHCcn6y_VhzbjrbwVr-Cymai20buKNuxsCJYlI/edit

3. Третій план — Ядро моделі та навчальний runtime
https://docs.google.com/document/d/1PizqJqX8q6xTJyF2DutYeXRelgPf8qQl-yioK9_EXjA/edit

4. Четвертий план — Оцінювання, inference та serving
https://docs.google.com/document/d/1A4_vrQSSiNQRLH4jVC5ySXxEPOt7PtWoaisvoAf328E/edit

5. П’ятий план — Agent Runtime, пам’ять та інструменти
https://docs.google.com/document/d/1LuE1aMPdRMgIJTSg-7eXXq1iGnYRXG6LWSVOPRmeuy0/edit

6. Шостий план — Самонавчання, research та Evolution Engine
https://docs.google.com/document/d/1LzVPyRV8lA1kCkVN1RRiEgIzQTINvQvGH2YN-jacXxk/edit

7. Сьомий план — Масштабування та distributed infrastructure
https://docs.google.com/document/d/1sWkveDm1OBkYh_pK0UMRSVFrjHJWHRGwMzLyH_6sZXc/edit

8. Восьмий план — Qualification, evidence та autonomous repair fabric
https://docs.google.com/document/d/10rhE3yzfgMaaapxwfpoMv1smWeXLzrBZnq22yEa87lY/edit

9. Дев’ятий план — Реальні навчальні кампанії та champion lifecycle
https://docs.google.com/document/d/1OL7-hhg-5jC70mWOLbN-X_flGZ2A8GL9HEp24ZJClns/edit

10. Десятий план — Product integration, Nika, Windows, cloud та release
https://docs.google.com/document/d/1vDUREMQTmhdyQDzgThphHw9jysLdgiPO52rt_JATUFE/edit

## Worker selection rule

When the owner assigns a numbered plan, read that plan and live GitHub.
Skip every Section marked DONE and start the numerically first unfinished Section **inside that assigned plan**.
Do not use legacy monolithic Section numbers to choose work.

Plans 1–8 may be worked and completed in any order and in parallel.
Live closure status comes from `MULTI_PLAN_CLOSURE_STATE.md`; Drive status lines are snapshots.
Cross-plan development uses Migration Contract Baseline v1 and versioned contracts/fixtures/mocks where the other implementation is not yet terminal; repository mutations obey `MULTI_PLAN_PARALLELISM_CONTRACT.md`.
Do not claim whole-product integration from a mock; whole-product convergence belongs to Plan 10.

Plan 9 starts only when the specific required terminal artifacts from upstream engineering plans exist.
Plan 10 is final convergence and release.

## Migrated accepted legacy work

- New Plan 1 / Section 1 = former Section 0 — DONE.
- New Plan 1 / Section 2 = former Section 1 — DONE.
- New Plan 8 / Section 1 = former Section 2 — DONE.
- New Plan 8 / Section 2 = former Section 3 — DONE.
- New Plan 8 / Section 3 = former Section 4 — QUALIFYING, not DONE at migration time.
