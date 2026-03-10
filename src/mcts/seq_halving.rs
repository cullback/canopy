/// Pre-computed Sequential Halving schedule.
///
/// Encodes the full simulation assignment for Gumbel Sequential Halving
/// as a lookup table, following the algorithm from DeepMind's mctx reference
/// implementation. At runtime the MCTS loop indexes into the schedule with
/// a monotonic simulation counter — no per-simulation arithmetic needed.
pub struct Schedule {
    /// `assignments[sim_index]` → candidate offset (0-indexed into the
    /// current candidate list) to force on that simulation.
    assignments: Vec<u16>,

    /// Sorted simulation indices at which the candidate set should be halved.
    halving_points: Vec<usize>,
}

impl Schedule {
    /// Build a schedule for `num_simulations` budget across `num_candidates`
    /// initial Gumbel-Top-k candidates.
    ///
    /// The algorithm mirrors mctx's `get_sequence_of_considered_visits`:
    /// each phase allocates `max(1, n / (log2(m) * considered))` full cycles
    /// through the current candidates, then halves the candidate count
    /// (minimum 2). Instead of returning considered-visit levels (which
    /// require matching edge visit counts — incompatible with virtual loss),
    /// we emit candidate offsets and record halving-point indices.
    pub fn new(num_simulations: usize, num_candidates: usize) -> Self {
        let n = num_simulations;
        let m = num_candidates;

        if m <= 1 {
            return Self {
                assignments: vec![0; n],
                halving_points: Vec::new(),
            };
        }

        let log2m = (m as f64).log2().ceil() as usize;
        let mut assignments = Vec::with_capacity(n);
        let mut halving_points = Vec::new();
        let mut num_considered = m;

        while assignments.len() < n {
            let extra = (n / (log2m * num_considered)).max(1);
            for _ in 0..extra {
                for c in 0..num_considered {
                    assignments.push(c as u16);
                }
            }

            let old_considered = num_considered;
            num_considered = (num_considered / 2).max(2);

            // Record halving point when candidates actually decreased and
            // there are more simulations to run.
            if num_considered < old_considered && assignments.len() < n {
                halving_points.push(assignments.len().min(n));
            }
        }

        assignments.truncate(n);
        halving_points.retain(|&p| p < n);

        Self {
            assignments,
            halving_points,
        }
    }

    /// Total number of simulations in the schedule.
    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    /// Which candidate offset to assign for the given simulation index.
    pub fn candidate_offset(&self, sim_index: usize) -> usize {
        self.assignments[sim_index] as usize
    }

    /// Whether the candidate set should be halved after committing
    /// simulation `sim_index` (i.e. a phase boundary was just crossed).
    pub fn should_halve(&self, sim_index: usize) -> bool {
        self.halving_points.contains(&sim_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: convert a schedule to the mctx-style considered-visits
    /// sequence so we can compare against the reference output.
    fn to_considered_visits(sched: &Schedule) -> Vec<u32> {
        let mut visits_per_candidate = vec![0u32; sched.assignments.len()];
        let mut result = Vec::with_capacity(sched.len());
        for &offset in &sched.assignments {
            let level = visits_per_candidate[offset as usize];
            result.push(level);
            visits_per_candidate[offset as usize] += 1;
        }
        result
    }

    /// Reference: run mctx's algorithm directly in Rust for comparison.
    fn mctx_reference(num_simulations: usize, max_candidates: usize) -> Vec<u32> {
        if max_candidates <= 1 {
            return (0..num_simulations as u32).collect();
        }
        let log2m = (max_candidates as f64).log2().ceil() as usize;
        let mut sequence = Vec::new();
        let mut visits = vec![0u32; max_candidates];
        let mut num_considered = max_candidates;
        while sequence.len() < num_simulations {
            let extra = (num_simulations / (log2m * num_considered)).max(1);
            for _ in 0..extra {
                sequence.extend_from_slice(&visits[..num_considered]);
                for v in visits[..num_considered].iter_mut() {
                    *v += 1;
                }
            }
            num_considered = (num_considered / 2).max(2);
        }
        sequence.truncate(num_simulations);
        sequence
    }

    #[test]
    fn matches_mctx_m4_n16() {
        let sched = Schedule::new(16, 4);
        assert_eq!(to_considered_visits(&sched), mctx_reference(16, 4));
    }

    #[test]
    fn matches_mctx_m8_n100() {
        let sched = Schedule::new(100, 8);
        assert_eq!(to_considered_visits(&sched), mctx_reference(100, 8));
    }

    #[test]
    fn matches_mctx_m16_n800() {
        let sched = Schedule::new(800, 16);
        assert_eq!(to_considered_visits(&sched), mctx_reference(800, 16));
    }

    #[test]
    fn matches_mctx_m2_n10() {
        let sched = Schedule::new(10, 2);
        assert_eq!(to_considered_visits(&sched), mctx_reference(10, 2));
    }

    #[test]
    fn single_candidate() {
        let sched = Schedule::new(10, 1);
        assert_eq!(sched.len(), 10);
        for i in 0..10 {
            assert_eq!(sched.candidate_offset(i), 0);
        }
        assert!(sched.halving_points.is_empty());
    }

    #[test]
    fn halving_points_correct_m4_n16() {
        let sched = Schedule::new(16, 4);
        // Phase 0: 4 candidates × 2 extra = 8 sims → halve at index 8
        assert!(sched.should_halve(8));
        // No other halving points (only 2 phases for m=4)
        assert!(!sched.should_halve(0));
        assert!(!sched.should_halve(4));
        assert!(!sched.should_halve(16));
    }

    #[test]
    fn schedule_length_matches_budget() {
        for &(n, m) in &[(16, 4), (100, 8), (800, 16), (10, 2), (50, 3)] {
            let sched = Schedule::new(n, m);
            assert_eq!(sched.len(), n, "n={n}, m={m}");
        }
    }

    #[test]
    fn offsets_within_bounds() {
        let sched = Schedule::new(800, 16);
        let mut max_offset = 0u16;
        for &a in &sched.assignments {
            max_offset = max_offset.max(a);
        }
        assert!(max_offset < 16);
    }
}
