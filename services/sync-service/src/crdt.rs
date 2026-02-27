//! CRDT Engine for conflict-free replicated data types
use bi_ide_protocol::sync::FileOperation;
use bi_ide_protocol::VectorClock;
use std::collections::HashMap;

/// CRDT Engine handles conflict resolution for file operations
pub struct CrdtEngine {
    /// Store of all operations by workspace
    operations: HashMap<String, Vec<FileOperation>>,
    /// Current vector clock by workspace
    vector_clocks: HashMap<String, VectorClock>,
}

impl CrdtEngine {
    pub fn new() -> Self {
        Self {
            operations: HashMap::new(),
            vector_clocks: HashMap::new(),
        }
    }

    /// Apply an operation to the CRDT
    /// Returns true if the operation was new and applied
    pub fn apply_operation(&mut self, op: &FileOperation) -> bool {
        let workspace_ops = self
            .operations
            .entry(op.workspace_id.clone())
            .or_insert_with(Vec::new);

        // Check if we've already seen this operation
        if workspace_ops.iter().any(|o| o.op_id == op.op_id) {
            return false;
        }

        // Apply the operation
        workspace_ops.push(op.clone());

        // Update vector clock
        let clock = self
            .vector_clocks
            .entry(op.workspace_id.clone())
            .or_insert_with(VectorClock::new);
        clock.merge(&op.vector_clock);
        clock.increment(op.node_id);

        true
    }

    /// Get all operations for a workspace
    pub fn get_operations(&self, workspace_id: &str) -> Vec<FileOperation> {
        self.operations
            .get(workspace_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Get operations since a given vector clock
    pub fn get_operations_since(
        &self,
        workspace_id: &str,
        since: &VectorClock,
    ) -> Vec<FileOperation> {
        let all_ops = self.operations.get(workspace_id);
        
        if let Some(ops) = all_ops {
            ops.iter()
                .filter(|op| {
                    // Include if this operation is not covered by the since clock
                    since.get(op.node_id) < op.vector_clock.get(op.node_id)
                })
                .cloned()
                .collect()
        } else {
            vec![]
        }
    }

    /// Get current vector clock for workspace
    pub fn get_vector_clock(&self, workspace_id: &str) -> VectorClock {
        self.vector_clocks
            .get(workspace_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Merge two file contents (for text files)
    /// Uses a simple line-based merge algorithm
    pub fn merge_files(
        &self,
        local_content: &str,
        remote_content: &str,
        base_content: Option<&str>,
    ) -> Result<String, MergeError> {
        match base_content {
            Some(base) => {
                // Three-way merge
                self.three_way_merge(local_content, remote_content, base)
            }
            None => {
                // Two-way merge (just concatenate with conflict markers)
                Ok(self.two_way_merge(local_content, remote_content))
            }
        }
    }

    fn three_way_merge(
        &self,
        local: &str,
        remote: &str,
        base: &str,
    ) -> Result<String, MergeError> {
        // Simple line-based three-way merge
        let local_lines: Vec<_> = local.lines().collect();
        let remote_lines: Vec<_> = remote.lines().collect();
        let base_lines: Vec<_> = base.lines().collect();

        let mut result = Vec::new();
        let mut conflicts = 0;

        let max_len = local_lines.len().max(remote_lines.len()).max(base_lines.len());

        for i in 0..max_len {
            let local_line = local_lines.get(i);
            let remote_line = remote_lines.get(i);
            let base_line = base_lines.get(i);

            match (local_line, remote_line, base_line) {
                // Both same as base or both changed to same
                (Some(l), Some(r), Some(b)) if l == r => {
                    result.push(*l);
                }
                // Only local changed
                (Some(l), Some(r), Some(b)) if r == b => {
                    result.push(*l);
                }
                // Only remote changed
                (Some(l), Some(r), Some(b)) if l == b => {
                    result.push(*r);
                }
                // Both changed differently - conflict
                (Some(l), Some(r), Some(b)) => {
                    conflicts += 1;
                    result.push("<<<<<<< local");
                    result.push(*l);
                    result.push("=======");
                    result.push(*r);
                    result.push(">>>>>>> remote");
                }
                // Lines added/removed
                (Some(l), None, _) => result.push(*l),
                (None, Some(r), _) => result.push(*r),
                (None, None, _) => {}
                _ => {}
            }
        }

        if conflicts > 0 {
            Err(MergeError::Conflicts(result.join("\n")))
        } else {
            Ok(result.join("\n"))
        }
    }

    fn two_way_merge(&self, local: &str, remote: &str) -> String {
        format!(
            "<<<<<<< local\n{}\n=======\n{}\n>>>>>>> remote",
            local, remote
        )
    }

    /// Resolve a conflict by choosing one version
    pub fn resolve_conflict(
        &self,
        merged_content: &str,
        resolution: ConflictResolution,
    ) -> String {
        match resolution {
            ConflictResolution::UseLocal => {
                // Extract local version
                self.extract_version(merged_content, "local")
            }
            ConflictResolution::UseRemote => {
                // Extract remote version
                self.extract_version(merged_content, "remote")
            }
        }
    }

    fn extract_version(&self, content: &str, version: &str) -> String {
        let marker_start = format!("<<<<<<< {}", version);
        let marker_end = ">>>>>>> remote";
        let marker_sep = "=======";

        let mut result = Vec::new();
        let mut in_conflict = false;
        let mut use_this_version = false;

        for line in content.lines() {
            if line.starts_with("<<<<<<<") {
                in_conflict = true;
                use_this_version = line == marker_start;
            } else if line == marker_sep {
                use_this_version = false;
            } else if line.starts_with(">>>>>>>") {
                in_conflict = false;
                use_this_version = false;
            } else if !in_conflict || use_this_version {
                result.push(line);
            }
        }

        result.join("\n")
    }
}

#[derive(Debug)]
pub enum MergeError {
    Conflicts(String),
}

pub enum ConflictResolution {
    UseLocal,
    UseRemote,
}

impl Default for CrdtEngine {
    fn default() -> Self {
        Self::new()
    }
}
