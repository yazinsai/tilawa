# 16 Retire the scoreboard front page

Status: ready-for-agent
Type: task
Blocked by: 11

Spec: `../spec.md` decisions 1 and 21.

Archive `scoreboard.json` history read-only, move the release gates to the dashboard's release tab, and replace the AGENTS.md scoreboard rule with: "Any change that can move a metric is an experiment with a recorded result; fold-in updates the main line. See `.scratch/lab/spec.md`."
