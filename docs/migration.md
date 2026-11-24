# Moving callers to CNV_Model

Old services and ports: `CNV_MiniCPM52B_Model` 29200, `CNV_BitNet2B_Model` 29201, `CNV_LFM3_Model` 29202, `CNV_SparkX26_Model` 29203. `CNV_Model` uses 29204, so the old services can keep running while callers move.

Map: port 29200 → `minicpm52b`, 29201 → `bitnet2b`, 29202 → `lfm3`, 29203 → `sparkx26`. Callers add `"model": "<name>"` to `POST /v1/chat` and use port 29204. Tokens do not carry over (new token prefix `cnv_`).

Callers found by search (not yet changed):

| File | Old reference | New reference |
|---|---|---|
| `AI/ToolServices/Converse/src/app.py:147` | probe `http://127.0.0.1:29202` | `http://127.0.0.1:29204`, send `model` |
| `AI/ToolServices/Converse/src/app.py:146`, `docs/purpose.md`, `README.md` | text names `CNV_LFM3_Model` | `CNV_Model` |
| `AI/ToolServices/_docs/PURPOSES.md:11` | ConversationGateway needs `CNV_Min...` | `CNV_Model` |
| `Applications/ServiceHub/README.md`, `src/app.py:37` | example URL `:29202` | any (example only) |
| `AI/ModelServices/_docs/*` | per-service rows (PURPOSES, SHARED-FEATURES, SERVICE-SPECIFIC, HARDWARE, GUIDES, SERVICE-ICONS, DIAGRAM-PROMPTS) | one `CNV_Model` row; old rows kept until the old repos are retired |
| `Brain/Projects/MiniCPM52BModel.md`, `Brain/Notes/*ConversationGateway*` | old names | update with project-sync after the switch |

Not searched: `.env*` files (never read) and the Citadel/Legion config in the database or env of other stacks; check those by hand for `29200`–`29203`.
