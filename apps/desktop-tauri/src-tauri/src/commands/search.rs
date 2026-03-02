//! Search Commands using ripgrep
//! High-performance text search across workspace

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::process::Stdio;
use tokio::process::Command;
use tracing::{debug, error, info, warn};

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub workspace_path: String,
    pub options: SearchOptions,
}

#[derive(Debug, Deserialize)]
pub struct SearchOptions {
    pub case_sensitive: bool,
    pub whole_word: bool,
    pub use_regex: bool,
    pub include_pattern: Option<String>,
    pub exclude_pattern: Option<String>,
    pub max_results: usize,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub file_path: String,
    pub file_name: String,
    pub matches: Vec<MatchInfo>,
}

#[derive(Debug, Serialize)]
pub struct MatchInfo {
    pub line_number: u32,
    pub column: u32,
    pub line_text: String,
    pub match_text: String,
}

#[tauri::command]
pub async fn search_workspace(request: SearchRequest) -> Result<Vec<SearchResult>, String> {
    info!(
        "Searching workspace: {} for: {}",
        request.workspace_path, request.query
    );

    // Check if ripgrep is available
    let rg_check = Command::new("rg")
        .arg("--version")
        .output()
        .await;

    if rg_check.is_err() {
        warn!("ripgrep not found, falling back to grep");
        return fallback_search(request).await;
    }

    // Build ripgrep command
    let mut cmd = Command::new("rg");
    
    // Output format: file:line:column:text
    cmd.arg("--column")
        .arg("--line-number")
        .arg("--with-filename")
        .arg("--no-heading")
        .arg("--color=never")
        .arg("--max-count")
        .arg(request.options.max_results.to_string());

    // Options
    if request.options.case_sensitive {
        cmd.arg("--case-sensitive");
    } else {
        cmd.arg("--ignore-case");
    }

    if request.options.whole_word {
        cmd.arg("--word-regexp");
    }

    if request.options.use_regex {
        // ripgrep uses regex by default
    } else {
        cmd.arg("--fixed-strings");
    }

    // Include/exclude patterns
    if let Some(include) = &request.options.include_pattern {
        for pattern in include.split(',') {
            let pattern = pattern.trim();
            if !pattern.is_empty() {
                cmd.arg("--glob").arg(pattern);
            }
        }
    }

    if let Some(exclude) = &request.options.exclude_pattern {
        for pattern in exclude.split(',') {
            let pattern = pattern.trim();
            if !pattern.is_empty() {
                cmd.arg("--glob").arg(format!("!{}", pattern));
            }
        }
    }

    // Add query and path
    cmd.arg(&request.query)
        .arg(&request.workspace_path);

    debug!("Running ripgrep command: {:?}", cmd);

    let output = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await
        .map_err(|e| format!("Failed to execute search: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("No files were searched") {
            return Ok(vec![]);
        }
        // Non-zero exit can also mean no matches found
        if output.stdout.is_empty() {
            return Ok(vec![]);
        }
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let results = parse_ripgrep_output(&stdout, &request.workspace_path)?;
    
    info!("Search found {} results in {} files", 
        results.iter().map(|r| r.matches.len()).sum::<usize>(),
        results.len()
    );

    Ok(results)
}

fn parse_ripgrep_output(output: &str, _workspace_path: &str) -> Result<Vec<SearchResult>, String> {
    let mut file_matches: std::collections::HashMap<String, Vec<MatchInfo>> = 
        std::collections::HashMap::new();

    for line in output.lines() {
        if line.is_empty() {
            continue;
        }

        // Parse format: file:line:column:text
        let parts: Vec<&str> = line.splitn(4, ':').collect();
        if parts.len() < 4 {
            continue;
        }

        let file_path = parts[0].to_string();
        let line_number: u32 = parts[1].parse().unwrap_or(0);
        let column: u32 = parts[2].parse().unwrap_or(0);
        let line_text = parts[3].to_string();

        // Extract match text (this is approximate)
        let match_text = line_text.clone();

        let match_info = MatchInfo {
            line_number,
            column,
            line_text: line_text.trim().to_string(),
            match_text,
        };

        file_matches
            .entry(file_path)
            .or_default()
            .push(match_info);
    }

    let mut results: Vec<SearchResult> = file_matches
        .into_iter()
        .map(|(file_path, matches)| {
            let file_name = PathBuf::from(&file_path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&file_path)
                .to_string();

            SearchResult {
                file_path,
                file_name,
                matches,
            }
        })
        .collect();

    // Sort by file path
    results.sort_by(|a, b| a.file_path.cmp(&b.file_path));

    Ok(results)
}

async fn fallback_search(request: SearchRequest) -> Result<Vec<SearchResult>, String> {
    // Fallback implementation using find + grep
    info!("Using fallback search implementation");

    let mut cmd = Command::new("grep");
    
    cmd.arg("-r")
        .arg("-n")
        .arg("--with-filename");

    if !request.options.case_sensitive {
        cmd.arg("-i");
    }

    if request.options.whole_word {
        cmd.arg("-w");
    }

    cmd.arg(&request.query)
        .arg(&request.workspace_path);

    // Add excludes
    if let Some(exclude) = &request.options.exclude_pattern {
        for pattern in exclude.split(',') {
            let pattern = pattern.trim();
            if !pattern.is_empty() {
                cmd.arg(format!("--exclude-dir={}", pattern.trim_end_matches('/')));
            }
        }
    }

    let output = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .await
        .map_err(|e| format!("Failed to execute grep: {}", e))?;

    if !output.status.success() && output.stdout.is_empty() {
        return Ok(vec![]);
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut file_matches: std::collections::HashMap<String, Vec<MatchInfo>> = 
        std::collections::HashMap::new();

    for line in stdout.lines() {
        // Parse format: file:line:text
        let parts: Vec<&str> = line.splitn(3, ':').collect();
        if parts.len() < 3 {
            continue;
        }

        let file_path = parts[0].to_string();
        let line_number: u32 = parts[1].parse().unwrap_or(0);
        let line_text = parts[2].to_string();

        let match_info = MatchInfo {
            line_number,
            column: 0,
            line_text: line_text.trim().to_string(),
            match_text: request.query.clone(),
        };

        file_matches
            .entry(file_path)
            .or_default()
            .push(match_info);
    }

    let mut results: Vec<SearchResult> = file_matches
        .into_iter()
        .map(|(file_path, matches)| {
            let file_name = PathBuf::from(&file_path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&file_path)
                .to_string();

            SearchResult {
                file_path,
                file_name,
                matches,
            }
        })
        .collect();

    results.sort_by(|a, b| a.file_path.cmp(&b.file_path));

    Ok(results)
}

/// Replace text in a file
#[derive(Debug, Deserialize)]
pub struct ReplaceRequest {
    pub file_path: String,
    pub search_query: String,
    pub replace_query: String,
    pub options: ReplaceOptions,
}

#[derive(Debug, Deserialize)]
pub struct ReplaceOptions {
    pub case_sensitive: bool,
    pub whole_word: bool,
    pub use_regex: bool,
}

#[derive(Debug, Serialize)]
pub struct ReplaceResult {
    pub replacements_count: usize,
    pub file_path: String,
}

#[tauri::command]
pub async fn replace_in_file(request: ReplaceRequest) -> Result<ReplaceResult, String> {
    info!("Replacing in file: {}", request.file_path);

    // Read file content
    let content = tokio::fs::read_to_string(&request.file_path)
        .await
        .map_err(|e| format!("Failed to read file: {}", e))?;

    // Perform replacement
    let new_content = if request.options.use_regex {
        // Use regex for replacement
        let regex = regex::Regex::new(&request.search_query)
            .map_err(|e| format!("Invalid regex: {}", e))?;
        regex.replace_all(&content, &request.replace_query).to_string()
    } else {
        // Simple string replacement
        if request.options.case_sensitive {
            content.replace(&request.search_query, &request.replace_query)
        } else {
            // Case-insensitive replacement
            let mut result = content.clone();
            let search_lower = request.search_query.to_lowercase();
            let mut start = 0;
            
            while let Some(pos) = result[start..].to_lowercase().find(&search_lower) {
                let actual_pos = start + pos;
                result.replace_range(
                    actual_pos..actual_pos + request.search_query.len(),
                    &request.replace_query
                );
                start = actual_pos + request.replace_query.len();
            }
            result
        }
    };

    let replacements_count = content.lines().count() - new_content.lines().count();

    // Write back
    tokio::fs::write(&request.file_path, new_content)
        .await
        .map_err(|e| format!("Failed to write file: {}", e))?;

    Ok(ReplaceResult {
        replacements_count,
        file_path: request.file_path,
    })
}

/// Replace all occurrences across workspace
#[derive(Debug, Deserialize)]
pub struct ReplaceAllRequest {
    pub search_query: String,
    pub replace_query: String,
    #[allow(dead_code)]
    pub workspace_path: String,
    pub options: ReplaceOptions,
    pub file_results: Vec<String>, // Files to replace in
}

#[derive(Debug, Serialize)]
pub struct ReplaceAllResult {
    pub total_replacements: usize,
    pub files_modified: usize,
    pub errors: Vec<String>,
}

#[tauri::command]
pub async fn replace_all(request: ReplaceAllRequest) -> Result<ReplaceAllResult, String> {
    info!(
        "Replacing all occurrences of '{}' with '{}' in {} files",
        request.search_query,
        request.replace_query,
        request.file_results.len()
    );

    let mut total_replacements = 0;
    let mut files_modified = 0;
    let mut errors = vec![];

    for file_path in &request.file_results {
        let replace_request = ReplaceRequest {
            file_path: file_path.clone(),
            search_query: request.search_query.clone(),
            replace_query: request.replace_query.clone(),
            options: ReplaceOptions {
                case_sensitive: request.options.case_sensitive,
                whole_word: request.options.whole_word,
                use_regex: request.options.use_regex,
            },
        };

        match replace_in_file(replace_request).await {
            Ok(result) => {
                if result.replacements_count > 0 {
                    total_replacements += result.replacements_count;
                    files_modified += 1;
                }
            }
            Err(e) => {
                error!("Failed to replace in {}: {}", file_path, e);
                errors.push(format!("{}: {}", file_path, e));
            }
        }
    }

    Ok(ReplaceAllResult {
        total_replacements,
        files_modified,
        errors,
    })
}
