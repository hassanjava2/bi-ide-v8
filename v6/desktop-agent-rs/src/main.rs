use anyhow::{anyhow, Context, Result};
use clap::Parser;
use reqwest::blocking::Client;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::process::Command;
use std::thread;
use std::time::Duration;

#[derive(Parser, Debug)]
#[command(name = "bi-ide-desktop-agent")]
struct Args {
    #[arg(long, env = "ORCH_SERVER", default_value = "http://localhost:8000")]
    server: String,

    #[arg(long, env = "ORCH_TOKEN")]
    token: Option<String>,

    #[arg(long, env = "WORKER_NAME", default_value = "desktop-rs")]
    name: String,

    #[arg(long, env = "WORKER_LABELS", default_value = "desktop,autonomous")]
    labels: String,

    #[arg(long, env = "WORKER_POLL_SEC", default_value_t = 5)]
    poll_sec: u64,
}

#[derive(Deserialize)]
struct RegisterResponse {
    worker_id: String,
}

#[derive(Deserialize)]
struct ClaimResponse {
    job: Option<Job>,
}

#[derive(Deserialize, Debug)]
struct Job {
    id: String,
    name: String,
    command: String,
    shell: bool,
    args: Vec<String>,
    env: serde_json::Value,
    cwd: Option<String>,
}

#[derive(Serialize)]
struct JobStatusUpdate<'a> {
    status: &'a str,
    return_code: Option<i32>,
    logs_tail: Option<&'a str>,
    error: Option<&'a str>,
}

fn headers(token: &Option<String>) -> Result<HeaderMap> {
    let mut map = HeaderMap::new();
    map.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    if let Some(token_value) = token {
        map.insert(
            "X-Orchestrator-Token",
            HeaderValue::from_str(token_value).context("invalid token header")?,
        );
    }
    Ok(map)
}

fn register_worker(client: &Client, base: &str, args: &Args) -> Result<String> {
    let labels: Vec<String> = args
        .labels
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let payload = json!({
        "name": args.name,
        "platform": std::env::consts::OS,
        "hostname": hostname::get().map(|h| h.to_string_lossy().to_string()).unwrap_or_else(|_| "unknown".to_string()),
        "python_version": "n/a",
        "cpu_count": num_cpus::get(),
        "gpu": null,
        "labels": labels,
    });

    let response = client
        .post(format!("{}/api/v1/orchestrator/workers/register", base))
        .json(&payload)
        .send()
        .context("worker register request failed")?;

    if !response.status().is_success() {
        return Err(anyhow!("worker register failed: {}", response.status()));
    }

    let data: RegisterResponse = response.json().context("invalid register response")?;
    Ok(data.worker_id)
}

fn heartbeat(client: &Client, base: &str, worker_id: &str, current_job: Option<&str>) -> Result<bool> {
    let payload = json!({"current_job_id": current_job});
    let response = client
        .post(format!("{}/api/v1/orchestrator/workers/{}/heartbeat", base, worker_id))
        .json(&payload)
        .send()
        .context("heartbeat failed")?;

    if !response.status().is_success() {
        return Err(anyhow!("heartbeat status {}", response.status()));
    }

    let body: serde_json::Value = response.json().unwrap_or_else(|_| json!({}));
    Ok(body
        .get("stop_current_job")
        .and_then(|v| v.as_bool())
        .unwrap_or(false))
}

fn claim_job(client: &Client, base: &str, worker_id: &str) -> Result<Option<Job>> {
    let response = client
        .post(format!("{}/api/v1/orchestrator/workers/{}/jobs/next", base, worker_id))
        .json(&json!({}))
        .send()
        .context("claim failed")?;

    if !response.status().is_success() {
        return Err(anyhow!("claim status {}", response.status()));
    }

    let body: ClaimResponse = response.json().context("invalid claim response")?;
    Ok(body.job)
}

fn update_status(client: &Client, base: &str, job_id: &str, payload: &JobStatusUpdate<'_>) -> Result<()> {
    let response = client
        .post(format!("{}/api/v1/orchestrator/jobs/{}/status", base, job_id))
        .json(payload)
        .send()
        .context("status update failed")?;

    if !response.status().is_success() {
        return Err(anyhow!("status update HTTP {}", response.status()));
    }

    Ok(())
}

fn run_job(job: &Job) -> (i32, String, Option<String>) {
    let mut command = if cfg!(target_os = "windows") {
        let mut c = Command::new("powershell");
        c.arg("-NoProfile").arg("-Command");
        c
    } else {
        let mut c = Command::new("/bin/sh");
        c.arg("-lc");
        c
    };

    if let Some(cwd) = &job.cwd {
        command.current_dir(cwd);
    }

    if job.shell {
        command.arg(&job.command);
    } else {
        let joined = std::iter::once(job.command.clone())
            .chain(job.args.clone())
            .collect::<Vec<_>>()
            .join(" ");
        command.arg(joined);
    }

    let output = command.output();
    match output {
        Ok(result) => {
            let code = result.status.code().unwrap_or(1);
            let stdout = String::from_utf8_lossy(&result.stdout);
            let stderr = String::from_utf8_lossy(&result.stderr);
            let mut merged = format!("[STDOUT]\n{}\n[STDERR]\n{}", stdout, stderr);
            if merged.len() > 12_000 {
                merged = merged[merged.len() - 12_000..].to_string();
            }
            (code, merged, None)
        }
        Err(err) => (1, String::new(), Some(format!("process spawn error: {}", err))),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let base = args.server.trim_end_matches('/').to_string();

    let mut client_builder = Client::builder().timeout(Duration::from_secs(30));
    client_builder = client_builder.default_headers(headers(&args.token)?);
    let client = client_builder.build().context("failed to build HTTP client")?;

    let worker_id = register_worker(&client, &base, &args)?;
    println!("worker registered: {}", worker_id);

    loop {
        if let Err(err) = heartbeat(&client, &base, &worker_id, None) {
            eprintln!("heartbeat error: {}", err);
            thread::sleep(Duration::from_secs(args.poll_sec.max(2)));
            continue;
        }

        let job = match claim_job(&client, &base, &worker_id) {
            Ok(value) => value,
            Err(err) => {
                eprintln!("claim error: {}", err);
                thread::sleep(Duration::from_secs(args.poll_sec.max(2)));
                continue;
            }
        };

        if let Some(job) = job {
            println!("running job {} - {}", job.id, job.name);

            let _ = update_status(
                &client,
                &base,
                &job.id,
                &JobStatusUpdate {
                    status: "running",
                    return_code: None,
                    logs_tail: Some("job started by rust desktop agent"),
                    error: None,
                },
            );

            let (code, logs, error) = run_job(&job);
            let status = if code == 0 { "completed" } else { "failed" };

            let _ = update_status(
                &client,
                &base,
                &job.id,
                &JobStatusUpdate {
                    status,
                    return_code: Some(code),
                    logs_tail: Some(&logs),
                    error: error.as_deref(),
                },
            );

            let _ = heartbeat(&client, &base, &worker_id, None);
        }

        thread::sleep(Duration::from_secs(args.poll_sec.max(2)));
    }
}
