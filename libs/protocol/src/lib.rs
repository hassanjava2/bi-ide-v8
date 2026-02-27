//! BI-IDE Protocol Library
//! Shared contracts between desktop, sync service, and control plane

pub mod auth;
pub mod sync;
pub mod telemetry;
pub mod training;

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Generate timestamp in milliseconds
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Unique operation ID across all nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OpId {
    pub node_id: u64,
    pub logical_clock: u64,
}

impl OpId {
    pub fn new(node_id: u64, logical_clock: u64) -> Self {
        Self {
            node_id,
            logical_clock,
        }
    }

    pub fn to_string(&self) -> String {
        format!("{}:{}", self.node_id, self.logical_clock)
    }
}

impl std::fmt::Display for OpId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.node_id, self.logical_clock)
    }
}

/// Vector clock for tracking causality
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct VectorClock {
    pub clocks: std::collections::HashMap<u64, u64>,
}

impl VectorClock {
    pub fn new() -> Self {
        Self {
            clocks: std::collections::HashMap::new(),
        }
    }

    pub fn increment(&mut self, node_id: u64) {
        let counter = self.clocks.entry(node_id).or_insert(0);
        *counter += 1;
    }

    pub fn get(&self, node_id: u64) -> u64 {
        self.clocks.get(&node_id).copied().unwrap_or(0)
    }

    pub fn merge(&mut self, other: &VectorClock) {
        for (node_id, clock) in &other.clocks {
            let entry = self.clocks.entry(*node_id).or_insert(0);
            *entry = (*entry).max(*clock);
        }
    }

    /// Compare two vector clocks
    /// Returns: Some(true) if self happens before other,
    ///          Some(false) if other happens before self,
    ///          None if concurrent
    pub fn compare(&self, other: &VectorClock) -> Option<bool> {
        let mut self_less = false;
        let mut other_less = false;

        let all_nodes: std::collections::HashSet<_> = self
            .clocks
            .keys()
            .chain(other.clocks.keys())
            .copied()
            .collect();

        for node_id in all_nodes {
            let self_clock = self.get(node_id);
            let other_clock = other.get(node_id);

            if self_clock < other_clock {
                self_less = true;
            } else if other_clock < self_clock {
                other_less = true;
            }
        }

        match (self_less, other_less) {
            (true, false) => Some(true),   // self happens before other
            (false, true) => Some(false),  // other happens before self
            (false, false) => Some(true),  // equal
            (true, true) => None,          // concurrent
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_clock_increment() {
        let mut vc = VectorClock::new();
        vc.increment(1);
        vc.increment(1);
        vc.increment(2);
        
        assert_eq!(vc.get(1), 2);
        assert_eq!(vc.get(2), 1);
        assert_eq!(vc.get(3), 0);
    }

    #[test]
    fn test_vector_clock_merge() {
        let mut vc1 = VectorClock::new();
        vc1.increment(1);
        vc1.increment(1);
        
        let mut vc2 = VectorClock::new();
        vc2.increment(2);
        vc2.increment(2);
        
        vc1.merge(&vc2);
        
        assert_eq!(vc1.get(1), 2);
        assert_eq!(vc1.get(2), 2);
    }

    #[test]
    fn test_vector_clock_compare() {
        let mut vc1 = VectorClock::new();
        vc1.increment(1);
        
        let mut vc2 = VectorClock::new();
        vc2.increment(1);
        vc2.increment(1);
        
        assert_eq!(vc1.compare(&vc2), Some(true));  // vc1 happens before vc2
        assert_eq!(vc2.compare(&vc1), Some(false)); // vc2 happens after vc1
    }
}
