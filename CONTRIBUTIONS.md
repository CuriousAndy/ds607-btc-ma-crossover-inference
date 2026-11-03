# Contributions

| Member | Student ID | Role | Files owned |
| --- | --- | --- | --- |
| Zhang Enhao | 01550648 | Group leader: topic and research design, topic clearance with the instructor, proposal, pre-registered parameters, repository architecture and module interfaces, task plan and timeline, node-by-node review and final audit, README (motivation, results, limitations), slides integration, rehearsal, discussant, submission | `src/config.py`, module interfaces in `src/`, `docs/proposal.pdf`, `README.md`, `AI_USAGE.md`, `CONTRIBUTIONS.md`, `requirements.txt`, `outputs/presentation/` |
| Deng Shenbojia | 01560703 | Data, trading rule, descriptive statistics, moving block bootstrap | `src/data.py`, `src/strategy.py`, `src/bootstrap.py`, `notebooks/01_data_rule_eda.ipynb`, `notebooks/03_block_bootstrap.ipynb` |
| Ma Tianjiao | 01564086 | Classical t-test baseline, Monte Carlo coverage check, robustness | `src/classical.py`, `notebooks/02_classical_baseline.ipynb`, `notebooks/04_monte_carlo_coverage.ipynb`, `notebooks/05_robustness.ipynb` |

## Notes

- Work was re-allocated once, on 21 Oct 2026: the block bootstrap moved from the group leader to Deng Shenbojia, and the robustness notebook (05) moved to Ma Tianjiao.
- Shared code: notebooks 03, 04 and 05 call Ma's `t_test_mean()` and Deng's `mbb_ci()` / `mbb_test()`. Ma first ran 04 and 05 with a temporary local bootstrap until Deng's `src/bootstrap.py` was merged; the results were identical, and the final notebooks use the shared function. Deng's notebook 03 uses Ma's `variance_factor()` as a cross-check.
- Review: the group leader re-ran every node from a clean kernel when it was submitted (logged in AI_USAGE.md) and ran a final clean-environment reproduction on 3 Nov 2026.
- Slides: each member built their own pages in a shared deck; the group leader wrote the opening, takeaway and limitations pages and merged the final PDF.
