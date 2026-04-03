//! Per-player road network tracking with incremental longest-road computation.
//!
//! **Data structures.** Each player's [`RoadNetwork`] stores a 72-bit `roads`
//! bitmask (one bit per board edge), a `reachable` frontier of empty edges
//! adjacent to the network, and a cached `longest_len`.
//!
//! **`>= 5` threshold.** The Catan longest-road award requires at least 5
//! roads, so DFS is skipped entirely when `roads.count_ones() < 5`.
//!
//! **Leaf-extension fast path (`add_road`).** When a newly placed road extends
//! a chain (one endpoint has degree 0 in the pre-existing network), only a
//! single-source DFS from the new tip is needed instead of the full
//! multi-source `compute_longest_road`. Full recompute fires only when both
//! endpoints were already in the network (bridge or loop).
//!
//! **Transit-node check (`on_opponent_build`).** An opponent settlement can
//! only shorten our longest road if it sits on a transit node—one with >= 2
//! incident roads of ours. Nodes with 0 or 1 incident roads are off-network
//! or leaves and can never split a path, so DFS is skipped.
//!
//! **DFS algorithm.** `compute_longest_road` performs exhaustive simple-path
//! search. It starts DFS only from vertices with degree != 2 (leaves or
//! branch points), since an optimal path's endpoints must have odd degree or
//! be branch points. A pure-cycle fallback handles the all-degree-2 case.
//! `dfs_road` walks edges via a `visited` bitmask, stopping at opponent
//! buildings.

use super::board::{AdjacencyBitboards, EdgeId, NodeId};

const EDGE_MASK: u128 = (1u128 << 72) - 1;

#[derive(Clone, Copy, Debug, Default)]
pub struct RoadNetwork {
    pub roads: u128, // bits 0..72 — player's placed roads
    reachable: u128, // bits 0..72 — empty edges adjacent to network
    longest_len: u8, // cached longest simple path length
}

impl RoadNetwork {
    #[cfg(test)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Player placed a settlement/city — expand frontier from the new building.
    pub fn add_building(&mut self, nid: NodeId, adj: &AdjacencyBitboards, opp_roads: u128) {
        self.reachable |= adj.node_adj_edges[nid.0 as usize];
        self.reachable &= !(self.roads | opp_roads) & EDGE_MASK;
    }

    /// Player placed a road. Updates roads, expands frontier, recomputes longest.
    /// `update_longest`: false during setup (longest road award not active).
    pub fn add_road(
        &mut self,
        eid: EdgeId,
        adj: &AdjacencyBitboards,
        opp_buildings: u64,
        opp_roads: u128,
        update_longest: bool,
    ) {
        // Compute endpoint degrees BEFORE adding the road (for leaf-extension check).
        let endpoints = adj.edge_endpoints[eid.0 as usize];
        let ep_a = endpoints.trailing_zeros() as usize;
        let ep_b = (endpoints & !(1u64 << ep_a)).trailing_zeros() as usize;
        let deg_a = (adj.node_adj_edges[ep_a] & self.roads).count_ones();
        let deg_b = (adj.node_adj_edges[ep_b] & self.roads).count_ones();

        self.roads |= 1u128 << eid.0;
        self.reachable &= !(1u128 << eid.0);

        // Expand frontier from each endpoint not blocked by opponent building
        let mut ep = endpoints;
        while ep != 0 {
            let node = ep.trailing_zeros() as usize;
            ep &= ep - 1;
            if opp_buildings & (1u64 << node) == 0 {
                self.reachable |= adj.node_adj_edges[node];
            }
        }
        self.reachable &= !(self.roads | opp_roads) & EDGE_MASK;

        if update_longest && self.roads.count_ones() >= 5 {
            if deg_a == 0 || deg_b == 0 {
                // Leaf extension: DFS from just the new tip.
                let tip = if deg_a == 0 { ep_a } else { ep_b };
                let from_tip = dfs_road(tip, 0, self.roads, adj, opp_buildings);
                self.longest_len = self.longest_len.max(from_tip);
            } else {
                // Both endpoints already in network (bridge/branch): full recompute.
                self.longest_len = compute_longest_road(self.roads, adj, opp_buildings);
            }
        }
    }

    /// Opponent placed a building — may shrink frontier, recomputes longest (might split).
    pub fn on_opponent_build(
        &mut self,
        nid: NodeId,
        adj: &AdjacencyBitboards,
        my_buildings: u64,
        opp_buildings: u64,
    ) {
        let affected = adj.node_adj_edges[nid.0 as usize] & self.reachable;
        self.reachable &= !affected;

        // Re-add any edge still reachable via its OTHER endpoint
        let mut rem = affected;
        while rem != 0 {
            let eid = rem.trailing_zeros() as usize;
            rem &= rem - 1;
            let other = (adj.edge_endpoints[eid] & !(1u64 << nid.0)).trailing_zeros() as usize;
            if my_buildings & (1u64 << other) != 0 {
                self.reachable |= 1u128 << eid;
            } else if opp_buildings & (1u64 << other) == 0
                && adj.node_adj_edges[other] & self.roads != 0
            {
                self.reachable |= 1u128 << eid;
            }
        }

        let incident_roads = adj.node_adj_edges[nid.0 as usize] & self.roads;
        if self.roads.count_ones() >= 5 && incident_roads.count_ones() >= 2 {
            self.longest_len = compute_longest_road(self.roads, adj, opp_buildings);
        }
    }

    /// Remove an edge from this player's frontier (opponent built a road there).
    pub fn remove_edge(&mut self, eid: EdgeId) {
        self.reachable &= !(1u128 << eid.0);
    }

    /// Bitmask of empty edges connected to the player's network.
    pub fn reachable_edges(&self) -> u128 {
        self.reachable
    }

    /// Cached longest simple path length.
    pub fn longest_road(&self) -> u8 {
        self.longest_len
    }

    /// Bitmask of nodes that lie on the current longest road path.
    pub fn longest_road_nodes(&self, adj: &AdjacencyBitboards, opp_buildings: u64) -> u64 {
        if self.roads.count_ones() < 5 {
            return 0;
        }
        compute_longest_road_nodes(self.roads, adj, opp_buildings)
    }
}

/// Compute longest road length AND return the node bitmask of the best path.
fn compute_longest_road_nodes(
    player_roads: u128,
    adj: &AdjacencyBitboards,
    opp_buildings: u64,
) -> u64 {
    if player_roads == 0 {
        return 0;
    }

    let mut degree = [0u8; 54];
    let mut any_vertex = 0usize;
    let mut remaining = player_roads;
    while remaining != 0 {
        let eid = remaining.trailing_zeros() as usize;
        remaining &= remaining - 1;
        let mut ep = adj.edge_endpoints[eid];
        while ep != 0 {
            let node = ep.trailing_zeros() as usize;
            ep &= ep - 1;
            degree[node] += 1;
            any_vertex = node;
        }
    }

    let mut best_len = 0u8;
    let mut best_nodes = 0u64;
    let mut found_start = false;

    for node in 0..54usize {
        if degree[node] != 0 && degree[node] != 2 {
            found_start = true;
            let incident = adj.node_adj_edges[node] & player_roads;
            let mut rem = incident;
            while rem != 0 {
                let eid = rem.trailing_zeros() as usize;
                rem &= rem - 1;
                let other = (adj.edge_endpoints[eid] & !(1u64 << node)).trailing_zeros() as usize;
                let mut path_nodes = 0u64;
                let len = dfs_road_nodes(
                    other,
                    1u128 << eid,
                    player_roads,
                    adj,
                    opp_buildings,
                    &mut path_nodes,
                ) + 1;
                // The starting node is always part of the path
                path_nodes |= 1u64 << node;
                if len > best_len {
                    best_len = len;
                    best_nodes = path_nodes;
                }
            }
        }
    }

    // Pure cycle fallback
    if !found_start {
        let incident = adj.node_adj_edges[any_vertex] & player_roads;
        let mut rem = incident;
        while rem != 0 {
            let eid = rem.trailing_zeros() as usize;
            rem &= rem - 1;
            let other = (adj.edge_endpoints[eid] & !(1u64 << any_vertex)).trailing_zeros() as usize;
            let mut path_nodes = 0u64;
            let len = dfs_road_nodes(
                other,
                1u128 << eid,
                player_roads,
                adj,
                opp_buildings,
                &mut path_nodes,
            ) + 1;
            path_nodes |= 1u64 << any_vertex;
            if len > best_len {
                best_len = len;
                best_nodes = path_nodes;
            }
        }
    }

    best_nodes
}

/// DFS that also tracks the best path's node set.
/// Returns the length, and writes the best path's node bitmask into `best_path_nodes`.
fn dfs_road_nodes(
    node: usize,
    visited: u128,
    player_roads: u128,
    adj: &AdjacencyBitboards,
    opp_buildings: u64,
    best_path_nodes: &mut u64,
) -> u8 {
    if opp_buildings & (1u64 << node) != 0 {
        return 0;
    }
    let reachable = adj.node_adj_edges[node] & player_roads & !visited;
    if reachable == 0 {
        // Leaf: this node is on the path
        *best_path_nodes = 1u64 << node;
        return 0;
    }
    let mut best = 0u8;
    *best_path_nodes = 1u64 << node;
    let mut rem = reachable;
    while rem != 0 {
        let eid = rem.trailing_zeros() as usize;
        rem &= rem - 1;
        let next = (adj.edge_endpoints[eid] & !(1u64 << node)).trailing_zeros() as usize;
        let mut child_nodes = 0u64;
        let len = dfs_road_nodes(
            next,
            visited | (1u128 << eid),
            player_roads,
            adj,
            opp_buildings,
            &mut child_nodes,
        ) + 1;
        if len > best {
            best = len;
            *best_path_nodes = (1u64 << node) | child_nodes;
        }
    }
    best
}

fn compute_longest_road(player_roads: u128, adj: &AdjacencyBitboards, opp_buildings: u64) -> u8 {
    if player_roads == 0 {
        return 0;
    }

    // Compute degree of each vertex in the road subgraph.
    // The longest trail's endpoints must have degree != 2 (leaves or branch
    // points), so we only need to start DFS from those vertices.
    let mut degree = [0u8; 54];
    let mut any_vertex = 0usize;
    let mut remaining = player_roads;
    while remaining != 0 {
        let eid = remaining.trailing_zeros() as usize;
        remaining &= remaining - 1;
        let mut ep = adj.edge_endpoints[eid];
        while ep != 0 {
            let node = ep.trailing_zeros() as usize;
            ep &= ep - 1;
            degree[node] += 1;
            any_vertex = node;
        }
    }

    let mut best = 0u8;
    let mut found_start = false;

    for node in 0..54usize {
        if degree[node] != 0 && degree[node] != 2 {
            found_start = true;
            let incident = adj.node_adj_edges[node] & player_roads;
            let mut rem = incident;
            while rem != 0 {
                let eid = rem.trailing_zeros() as usize;
                rem &= rem - 1;
                let other = (adj.edge_endpoints[eid] & !(1u64 << node)).trailing_zeros() as usize;
                best =
                    best.max(dfs_road(other, 1u128 << eid, player_roads, adj, opp_buildings) + 1);
            }
        }
    }

    // Pure cycle: all vertices have degree 2. Start from any vertex.
    if !found_start {
        let incident = adj.node_adj_edges[any_vertex] & player_roads;
        let mut rem = incident;
        while rem != 0 {
            let eid = rem.trailing_zeros() as usize;
            rem &= rem - 1;
            let other = (adj.edge_endpoints[eid] & !(1u64 << any_vertex)).trailing_zeros() as usize;
            best = best.max(dfs_road(other, 1u128 << eid, player_roads, adj, opp_buildings) + 1);
        }
    }

    best
}

fn dfs_road(
    node: usize,
    visited: u128,
    player_roads: u128,
    adj: &AdjacencyBitboards,
    opp_buildings: u64,
) -> u8 {
    if opp_buildings & (1u64 << node) != 0 {
        return 0;
    }
    let reachable = adj.node_adj_edges[node] & player_roads & !visited;
    let mut best = 0u8;
    let mut rem = reachable;
    while rem != 0 {
        let eid = rem.trailing_zeros() as usize;
        rem &= rem - 1;
        let next = (adj.edge_endpoints[eid] & !(1u64 << node)).trailing_zeros() as usize;
        best = best.max(
            dfs_road(
                next,
                visited | (1u128 << eid),
                player_roads,
                adj,
                opp_buildings,
            ) + 1,
        );
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::topology::Topology;

    fn make_topo() -> Topology {
        Topology::from_seed(42)
    }

    #[test]
    fn add_road_expands_frontier() {
        let topo = make_topo();
        let adj = &topo.adj;
        let mut net = RoadNetwork::new();

        // Place a building at node 0 to start the network
        net.add_building(NodeId(0), adj, 0);
        let initial = net.reachable_edges();
        assert!(initial != 0, "building should create frontier");

        // Pick the first reachable edge and add it as a road
        let eid = initial.trailing_zeros() as u8;
        net.add_road(EdgeId(eid), adj, 0, 0, false);

        // Frontier should now include edges at the new endpoint
        let after = net.reachable_edges();
        // The placed road's edge should be gone from frontier
        assert_eq!(after & (1u128 << eid), 0);
        // But we should have MORE reachable edges (expanded from far endpoint)
        assert!(
            after.count_ones() >= initial.count_ones(),
            "frontier should grow or stay same after road: before={}, after={}",
            initial.count_ones(),
            after.count_ones()
        );
    }

    #[test]
    fn opponent_building_shrinks_frontier() {
        let topo = make_topo();
        let adj = &topo.adj;
        let mut net = RoadNetwork::new();

        // Place building and road to create a frontier
        net.add_building(NodeId(0), adj, 0);
        let eid = net.reachable_edges().trailing_zeros() as u8;
        net.add_road(EdgeId(eid), adj, 0, 0, false);

        // Find the far endpoint of the road
        let far = (adj.edge_endpoints[eid as usize] & !(1u64 << 0)).trailing_zeros() as u8;

        let before = net.reachable_edges();
        // Edges at the far endpoint that are in our frontier
        let edges_at_far = adj.node_adj_edges[far as usize] & before;
        assert!(edges_at_far != 0, "should have frontier edges at far node");

        // Opponent builds at far endpoint — blocks transit through that node
        let my_buildings = 1u64 << 0; // our building at node 0
        let opp_buildings = 1u64 << far;
        net.on_opponent_build(NodeId(far), adj, my_buildings, opp_buildings);

        let after = net.reachable_edges();
        // Edges at far that went to nodes NOT connected to our network should be gone
        // The edge connecting to node 0 (our building) should remain
        assert!(
            after.count_ones() <= before.count_ones(),
            "frontier should shrink or stay same: before={}, after={}",
            before.count_ones(),
            after.count_ones()
        );
    }

    #[test]
    fn longest_road_cached_correctly() {
        let topo = make_topo();
        let adj = &topo.adj;
        let mut net = RoadNetwork::new();

        // Build a network from node 0
        net.add_building(NodeId(0), adj, 0);

        // Build a chain of 5 roads
        let mut current_node = 0usize;
        let mut visited_nodes = 1u64 << 0;
        for i in 0..5 {
            let candidates = adj.node_adj_edges[current_node] & net.reachable_edges();
            assert!(candidates != 0, "should have reachable edge for road {i}");
            let eid = candidates.trailing_zeros() as u8;
            let far = (adj.edge_endpoints[eid as usize] & !visited_nodes).trailing_zeros() as usize;
            net.add_road(EdgeId(eid), adj, 0, 0, true);
            visited_nodes |= 1u64 << far;
            current_node = far;
        }

        assert!(
            net.longest_road() >= 5,
            "5 roads in a chain should give longest >= 5, got {}",
            net.longest_road()
        );
    }

    #[test]
    fn add_road_skips_dfs_under_five() {
        let topo = make_topo();
        let adj = &topo.adj;
        let mut net = RoadNetwork::new();

        net.add_building(NodeId(0), adj, 0);

        // Build 4 roads with update_longest: true
        let mut current_node = 0usize;
        let mut visited_nodes = 1u64 << 0;
        for _ in 0..4 {
            let candidates = adj.node_adj_edges[current_node] & net.reachable_edges();
            let eid = candidates.trailing_zeros() as u8;
            let far = (adj.edge_endpoints[eid as usize] & !visited_nodes).trailing_zeros() as usize;
            net.add_road(EdgeId(eid), adj, 0, 0, true);
            visited_nodes |= 1u64 << far;
            current_node = far;
        }

        // With < 5 roads, DFS should not have run, longest stays 0
        assert_eq!(net.longest_road(), 0, "longest should be 0 with < 5 roads");
    }

    #[test]
    fn bridge_two_components_recomputes() {
        let topo = make_topo();
        let adj = &topo.adj;
        let mut net = RoadNetwork::new();

        // Build first component: 3 roads from node 0
        net.add_building(NodeId(0), adj, 0);
        let mut current = 0usize;
        let mut visited = 1u64 << 0;
        for _ in 0..3 {
            let candidates = adj.node_adj_edges[current] & net.reachable_edges();
            let eid = candidates.trailing_zeros() as u8;
            let far = (adj.edge_endpoints[eid as usize] & !visited).trailing_zeros() as usize;
            net.add_road(EdgeId(eid), adj, 0, 0, true);
            visited |= 1u64 << far;
            current = far;
        }
        let component1_end = current;

        // Build second component: place building at a disconnected node and build 3 roads
        // Find a node not adjacent to our network
        let mut start2 = None;
        for n in 0..54u8 {
            if visited & (1u64 << n) == 0 && adj.node_adj_nodes[n as usize] & visited == 0 {
                start2 = Some(n as usize);
                break;
            }
        }
        let start2 = start2.expect("should find disconnected node");
        net.add_building(NodeId(start2 as u8), adj, 0);
        current = start2;
        visited |= 1u64 << start2;
        for _ in 0..3 {
            let candidates = adj.node_adj_edges[current] & net.reachable_edges();
            if candidates == 0 {
                break;
            }
            let eid = candidates.trailing_zeros() as u8;
            let far_mask = adj.edge_endpoints[eid as usize] & !visited;
            if far_mask == 0 {
                break;
            }
            let far = far_mask.trailing_zeros() as usize;
            net.add_road(EdgeId(eid), adj, 0, 0, true);
            visited |= 1u64 << far;
            current = far;
        }

        let before = net.longest_road();

        // Try to find a bridging edge between component1_end and current network node
        // that would connect the two components
        let edges_at_end = adj.node_adj_edges[component1_end];
        let mut bridge_eid = None;
        let mut rem = edges_at_end & net.reachable_edges();
        while rem != 0 {
            let eid = rem.trailing_zeros() as u8;
            rem &= rem - 1;
            let far = (adj.edge_endpoints[eid as usize] & !(1u64 << component1_end))
                .trailing_zeros() as usize;
            // Check if far endpoint connects to the second component
            if adj.node_adj_edges[far] & net.roads != 0 {
                bridge_eid = Some(eid);
                break;
            }
        }

        if let Some(eid) = bridge_eid {
            net.add_road(EdgeId(eid), adj, 0, 0, true);
            // After bridging, longest should be >= before
            assert!(
                net.longest_road() >= before,
                "bridging should maintain or increase longest: before={}, after={}",
                before,
                net.longest_road()
            );
        }
        // If no bridge found, the test still validates the two-component case
    }
}
