//! Git operations wrapper
use anyhow::Result;
use std::path::Path;
use tokio::process::Command;

/// Execute git command in directory
pub async fn git_cmd(path: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(path)
        .args(args)
        .output()
        .await?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(anyhow::anyhow!("Git command failed: {}", stderr))
    }
}

/// Get repository status
pub async fn status(path: &Path) -> Result<GitStatus> {
    let output = git_cmd(path, &["status", "--porcelain", "-b"]).await?;
    
    let mut modified = Vec::new();
    let mut added = Vec::new();
    let mut deleted = Vec::new();
    let mut untracked = Vec::new();
    let mut renamed = Vec::new();
    
    let mut branch = String::from("main");
    let mut ahead = 0;
    let mut behind = 0;

    for line in output.lines() {
        if line.starts_with("##") {
            // Parse branch info
            let branch_info = &line[3..];
            if let Some(pos) = branch_info.find("...") {
                branch = branch_info[..pos].to_string();
            } else if let Some(pos) = branch_info.find(' ') {
                branch = branch_info[..pos].to_string();
            } else {
                branch = branch_info.to_string();
            }

            // Parse ahead/behind
            if let Some(start) = branch_info.find('[') {
                if let Some(end) = branch_info.find(']') {
                    let stats = &branch_info[start + 1..end];
                    for part in stats.split(',') {
                        let part = part.trim();
                        if part.starts_with("ahead ") {
                            ahead = part[6..].parse().unwrap_or(0);
                        } else if part.starts_with("behind ") {
                            behind = part[7..].parse().unwrap_or(0);
                        }
                    }
                }
            }
        } else if !line.is_empty() {
            let status = &line[..2];
            let file = &line[3..];

            match status {
                " M" | "M " | "MM" => modified.push(file.to_string()),
                " A" | "A " => added.push(file.to_string()),
                " D" | "D " => deleted.push(file.to_string()),
                "??" => untracked.push(file.to_string()),
                " R" | "R " => {
                    if let Some(pos) = file.find(" -> ") {
                        renamed.push((file[..pos].to_string(), file[pos + 4..].to_string()));
                    }
                }
                _ => {}
            }
        }
    }

    let is_clean = modified.is_empty() 
        && added.is_empty() 
        && deleted.is_empty() 
        && untracked.is_empty()
        && renamed.is_empty();

    Ok(GitStatus {
        branch,
        ahead,
        behind,
        modified,
        added,
        deleted,
        untracked,
        renamed,
        is_clean,
    })
}

/// Stage files
pub async fn add(path: &Path, files: &[String]) -> Result<()> {
    let mut args = vec!["add"];
    let files_refs: Vec<&str> = files.iter().map(|s| s.as_str()).collect();
    args.extend(files_refs);
    
    git_cmd(path, &args).await?;
    Ok(())
}

/// Commit changes
pub async fn commit(path: &Path, message: &str) -> Result<String> {
    git_cmd(path, &["commit", "-m", message]).await
}

/// Push to remote
pub async fn push(path: &Path, remote: &str, branch: &str) -> Result<()> {
    git_cmd(path, &["push", remote, branch]).await?;
    Ok(())
}

/// Pull from remote
pub async fn pull(path: &Path, remote: &str, branch: &str) -> Result<()> {
    git_cmd(path, &["pull", remote, branch]).await?;
    Ok(())
}

/// Get commit log
pub async fn log(path: &Path, limit: usize) -> Result<Vec<Commit>> {
    let format = "%H|%s|%an|%at";
    let output = git_cmd(
        path,
        &["log", &format!("--format={}", format), &format!("-n{}", limit)]
    ).await?;

    let mut commits = Vec::new();

    for line in output.lines() {
        let parts: Vec<_> = line.split('|').collect();
        if parts.len() >= 4 {
            commits.push(Commit {
                hash: parts[0].to_string(),
                message: parts[1].to_string(),
                author: parts[2].to_string(),
                timestamp: parts[3].parse().unwrap_or(0),
            });
        }
    }

    Ok(commits)
}

/// Get branches
pub async fn branches(path: &Path) -> Result<Vec<Branch>> {
    let output = git_cmd(path, &["branch", "-a", "--format=%(refname:short)|%(HEAD)"]).await?;

    let mut branches = Vec::new();

    for line in output.lines() {
        let parts: Vec<_> = line.split('|').collect();
        if parts.len() >= 2 {
            let name = parts[0].to_string();
            let is_current = parts[1] == "*";
            let is_remote = name.starts_with("remotes/");

            branches.push(Branch {
                name,
                is_current,
                is_remote,
            });
        }
    }

    Ok(branches)
}

/// Checkout branch
pub async fn checkout(path: &Path, branch: &str, create: bool) -> Result<()> {
    if create {
        git_cmd(path, &["checkout", "-b", branch]).await?;
    } else {
        git_cmd(path, &["checkout", branch]).await?;
    }
    Ok(())
}

/// Clone repository
pub async fn clone(url: &str, path: &Path, depth: Option<usize>) -> Result<()> {
    let mut args = vec!["clone"];
    let depth_arg;
    
    if let Some(d) = depth {
        depth_arg = d.to_string();
        args.push("--depth");
        args.push(&depth_arg);
    }
    
    args.push(url);
    args.push(path.to_str().unwrap());

    let output = Command::new("git")
        .args(&args)
        .output()
        .await?;

    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(anyhow::anyhow!("Clone failed: {}", stderr))
    }
}

/// Check if path is a git repository
pub async fn is_repo(path: &Path) -> bool {
    path.join(".git").exists()
}

#[derive(Debug, Clone)]
pub struct GitStatus {
    pub branch: String,
    pub ahead: usize,
    pub behind: usize,
    pub modified: Vec<String>,
    pub added: Vec<String>,
    pub deleted: Vec<String>,
    pub untracked: Vec<String>,
    pub renamed: Vec<(String, String)>,
    pub is_clean: bool,
}

#[derive(Debug, Clone)]
pub struct Commit {
    pub hash: String,
    pub message: String,
    pub author: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone)]
pub struct Branch {
    pub name: String,
    pub is_current: bool,
    pub is_remote: bool,
}
