## Ideas

- <https://lczero.org/dev/lc0/search/lc3/>
- <https://arxiv.org/html/2511.14220v1>

## Reanalyze

Keep all self-play positions forever. Periodically re-run search on old positions with the current network to refresh policy/value targets. Decouples data generation from target quality.

Current flow:

```
self-play -> samples -> replay buffer (capped) -> train
```

With reanalyze:

```
self-play -> position store (permanent, outcomes only)
                  |
                  v
            reanalyze worker (re-searches with latest net)
                  |
                  v
              sample pool -> train
```

Each training batch mixes fresh self-play samples with reanalyzed samples (Will used 50/50). Reanalyze searches can be cheaper than self-play searches since we only need the root policy/value, not a full game trajectory.

Key decisions:

- **Reanalyze budget**: how many sims per reanalyzed position (can be less than self-play)
- **Mix ratio**: fraction of each batch from reanalyze vs fresh self-play
- **Staleness**: how often to re-search the same position (diminishing returns as net improves less between iterations)
- **Storage**: position store is just features + game outcome Z, no search targets

Reference: MuZero paper Appendix H. Will's TakZero: 50M positions, 50/50 mix, 4x target reuse.

## Playout cap randomization

Prerequisite for reanalyze to be cost-effective. 75% of moves use small budget (e.g. 50 sims), 25% use full budget. Only full-search positions produce policy targets. All positions produce value targets. ~4x more value samples per unit of search compute. The cheap searches are also good reanalyze candidates since they're fast to re-search.
