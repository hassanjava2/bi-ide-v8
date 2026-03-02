//! Self-Improvement Dashboard with Gated Learning

import { useState, useEffect } from "react";
import {
  Brain, TrendingUp, Shield, CheckCircle, XCircle, Clock,
  Play, Pause, RotateCcw, AlertTriangle, FileCode, TestTube,
  GitBranch, Zap, Activity, BarChart3, X
} from "lucide-react";
import { useStore } from "../../lib/store";
import { invoke } from "@tauri-apps/api/core";

interface LearningJob {
  id: string;
  name: string;
  type: "code_analysis" | "pattern_learning" | "model_tuning";
  status: "proposed" | "sandbox_testing" | "evaluating" | "passed" | "failed" | "promoted";
  progress: number;
  proposed_at: number;
  sandbox_results?: SandboxResult;
  evaluation_score?: number;
  promotion_gate?: PromotionGate;
}

interface SandboxResult {
  tests_passed: number;
  tests_failed: number;
  coverage: number;
  performance_impact: number;
  errors: string[];
}

interface PromotionGate {
  test_coverage_min: number;
  performance_degradation_max: number;
  human_approval: boolean;
  auto_rollback: boolean;
}

interface LearningMetrics {
  total_jobs: number;
  passed_jobs: number;
  failed_jobs: number;
  pending_jobs: number;
  average_evaluation_score: number;
  last_cycle_at: number;
}

const GATES: Record<string, PromotionGate> = {
  conservative: {
    test_coverage_min: 90,
    performance_degradation_max: 5,
    human_approval: true,
    auto_rollback: true,
  },
  balanced: {
    test_coverage_min: 80,
    performance_degradation_max: 10,
    human_approval: false,
    auto_rollback: true,
  },
  aggressive: {
    test_coverage_min: 70,
    performance_degradation_max: 15,
    human_approval: false,
    auto_rollback: false,
  },
};

export function LearningDashboard() {
  const [jobs, setJobs] = useState<LearningJob[]>([]);
  const [metrics, setMetrics] = useState<LearningMetrics | null>(null);
  const [selectedJob, setSelectedJob] = useState<LearningJob | null>(null);
  const [activeGate, setActiveGate] = useState<keyof typeof GATES>("balanced");
  const [isLearningActive, setIsLearningActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadJobs();
    loadMetrics();
    const interval = setInterval(() => {
      loadJobs();
      loadMetrics();
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  const loadJobs = async () => {
    try {
      const result = await invoke<LearningJob[]>("get_learning_jobs");
      setJobs(result);
      setError(null);
    } catch (err) {
      setJobs([]);
      setError("Learning jobs API is unavailable");
    }
  };

  const loadMetrics = async () => {
    try {
      const result = await invoke<LearningMetrics>("get_learning_metrics");
      setMetrics(result);
      setError(null);
    } catch (err) {
      setMetrics(null);
      setError("Learning metrics API is unavailable");
    }
  };

  const startLearning = async () => {
    setLoading(true);
    try {
      await invoke("start_learning_cycle", { gate: activeGate });
      setIsLearningActive(true);
    } catch (err) {
      console.error("Failed to start learning:", err);
    } finally {
      setLoading(false);
    }
  };

  const approvePromotion = async (jobId: string) => {
    try {
      await invoke("approve_job_promotion", { job_id: jobId });
      loadJobs();
    } catch (err) {
      console.error("Failed to approve:", err);
    }
  };

  const rollbackJob = async (jobId: string) => {
    try {
      await invoke("rollback_job", { job_id: jobId });
      loadJobs();
    } catch (err) {
      console.error("Failed to rollback:", err);
    }
  };

  const getStatusColor = (status: string) => {
    const colors: Record<string, string> = {
      proposed: "text-blue-400",
      sandbox_testing: "text-yellow-400",
      evaluating: "text-purple-400",
      passed: "text-green-400",
      failed: "text-red-400",
      promoted: "text-primary-400",
    };
    return colors[status] || "text-dark-400";
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "promoted": return <CheckCircle className="w-4 h-4 text-green-400" />;
      case "failed": return <XCircle className="w-4 h-4 text-red-400" />;
      case "sandbox_testing": return <TestTube className="w-4 h-4 text-yellow-400" />;
      case "evaluating": return <Activity className="w-4 h-4 text-purple-400" />;
      default: return <Clock className="w-4 h-4 text-blue-400" />;
    }
  };

  return (
    <div className="h-full flex flex-col bg-dark-800 p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-primary-400" />
          <span className="font-medium">Self-Improvement</span>
        </div>
        <button
          onClick={startLearning}
          disabled={loading || isLearningActive}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-primary-600 hover:bg-primary-500 disabled:bg-dark-700 text-white text-sm rounded"
        >
          {isLearningActive ? (
            <>
              <Activity className="w-4 h-4 animate-pulse" />
              Running...
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Start Cycle
            </>
          )}
        </button>
      </div>

      {error && (
        <div className="mb-4 p-2 bg-red-900/20 border border-red-700 rounded text-xs text-red-300">
          {error}
        </div>
      )}

      {/* Metrics */}
      {metrics && (
        <div className="grid grid-cols-4 gap-2 mb-4">
          <MetricCard label="Total" value={metrics.total_jobs} icon={<Brain className="w-4 h-4" />} />
          <MetricCard label="Passed" value={metrics.passed_jobs} color="text-green-400" icon={<CheckCircle className="w-4 h-4" />} />
          <MetricCard label="Failed" value={metrics.failed_jobs} color="text-red-400" icon={<XCircle className="w-4 h-4" />} />
          <MetricCard label="Score" value={`${(metrics.average_evaluation_score * 100).toFixed(0)}%`} icon={<TrendingUp className="w-4 h-4" />} />
        </div>
      )}

      {/* Gate Selection */}
      <div className="mb-4 p-3 bg-dark-900 rounded border border-dark-700">
        <div className="text-xs text-dark-400 mb-2">Promotion Gate</div>
        <div className="flex gap-2">
          {(Object.keys(GATES) as Array<keyof typeof GATES>).map((gate) => (
            <button
              key={gate}
              onClick={() => setActiveGate(gate)}
              className={`flex-1 px-3 py-2 rounded text-sm capitalize transition-colors
                ${activeGate === gate 
                  ? "bg-primary-600 text-white" 
                  : "bg-dark-700 hover:bg-dark-600"
                }`}
            >
              {gate}
            </button>
          ))}
        </div>
        <div className="mt-2 text-xs text-dark-500">
          Test Coverage: ≥{GATES[activeGate].test_coverage_min}% • 
          Performance Impact: ≤{GATES[activeGate].performance_degradation_max}% •
          Human Approval: {GATES[activeGate].human_approval ? "Required" : "Auto"}
        </div>
      </div>

      {/* Jobs List */}
      <div className="flex-1 overflow-y-auto">
        <div className="space-y-2">
          {jobs.map((job) => (
            <div
              key={job.id}
              onClick={() => setSelectedJob(job)}
              className="p-3 bg-dark-900 rounded border border-dark-700 hover:border-dark-600 cursor-pointer"
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-2">
                  {getStatusIcon(job.status)}
                  <div>
                    <div className="text-sm font-medium">{job.name}</div>
                    <div className="text-xs text-dark-500 capitalize">{job.type.replace("_", " ")}</div>
                  </div>
                </div>
                <span className={`text-xs ${getStatusColor(job.status)}`}>
                  {job.status.replace("_", " ")}
                </span>
              </div>
              
              <div className="mt-2">
                <div className="h-1.5 bg-dark-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all ${
                      job.status === "failed" ? "bg-red-500" :
                      job.status === "promoted" ? "bg-green-500" :
                      "bg-primary-500"
                    }`}
                    style={{ width: `${job.progress}%` }}
                  />
                </div>
              </div>

              {job.evaluation_score !== undefined && (
                <div className="mt-2 text-xs text-dark-400">
                  Score: {(job.evaluation_score * 100).toFixed(1)}%
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Job Detail Modal */}
      {selectedJob && (
        <JobDetailModal
          job={selectedJob}
          gate={GATES[activeGate]}
          onApprove={() => approvePromotion(selectedJob.id)}
          onRollback={() => rollbackJob(selectedJob.id)}
          onClose={() => setSelectedJob(null)}
        />
      )}
    </div>
  );
}

function MetricCard({ label, value, color, icon }: { label: string; value: string | number; color?: string; icon: React.ReactNode }) {
  return (
    <div className="p-2 bg-dark-900 rounded border border-dark-700 text-center">
      <div className="text-dark-400 mb-1">{icon}</div>
      <div className={`text-xl font-bold ${color || "text-dark-100"}`}>{value}</div>
      <div className="text-xs text-dark-500">{label}</div>
    </div>
  );
}

interface JobDetailModalProps {
  job: LearningJob;
  gate: PromotionGate;
  onApprove: () => void;
  onRollback: () => void;
  onClose: () => void;
}

function JobDetailModal({ job, gate, onApprove, onRollback, onClose }: JobDetailModalProps) {
  const canPromote = job.status === "passed" && job.evaluation_score && job.evaluation_score >= gate.test_coverage_min / 100;
  const needsApproval = gate.human_approval && job.status === "passed";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="w-full max-w-lg bg-dark-800 rounded-lg shadow-xl border border-dark-600">
        <div className="flex items-center justify-between p-4 border-b border-dark-700">
          <h3 className="font-medium">{job.name}</h3>
          <button onClick={onClose} className="p-1 hover:bg-dark-700 rounded">
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-4 space-y-4">
          {/* Status */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-dark-400">Status:</span>
            <span className={`text-sm font-medium ${
              job.status === "promoted" ? "text-green-400" :
              job.status === "failed" ? "text-red-400" :
              "text-primary-400"
            }`}>
              {job.status.replace("_", " ").toUpperCase()}
            </span>
          </div>

          {/* Sandbox Results */}
          {job.sandbox_results && (
            <div className="p-3 bg-dark-900 rounded border border-dark-700">
              <h4 className="text-sm font-medium mb-2">Sandbox Test Results</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="flex justify-between">
                  <span className="text-dark-400">Tests Passed</span>
                  <span className="text-green-400">{job.sandbox_results.tests_passed}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dark-400">Tests Failed</span>
                  <span className={job.sandbox_results.tests_failed > 0 ? "text-red-400" : "text-dark-500"}>
                    {job.sandbox_results.tests_failed}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dark-400">Coverage</span>
                  <span className={job.sandbox_results.coverage >= gate.test_coverage_min ? "text-green-400" : "text-yellow-400"}>
                    {job.sandbox_results.coverage}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dark-400">Performance</span>
                  <span className={job.sandbox_results.performance_impact <= gate.performance_degradation_max ? "text-green-400" : "text-yellow-400"}>
                    {job.sandbox_results.performance_impact > 0 ? "+" : ""}{job.sandbox_results.performance_impact}%
                  </span>
                </div>
              </div>

              {job.sandbox_results.errors.length > 0 && (
                <div className="mt-2">
                  <div className="text-sm text-red-400">Errors:</div>
                  {job.sandbox_results.errors.map((error, idx) => (
                    <div key={idx} className="text-xs text-dark-400 mt-1">• {error}</div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Gate Check */}
          <div className="p-3 bg-dark-900 rounded border border-dark-700">
            <h4 className="text-sm font-medium mb-2">Gate Requirements</h4>
            <div className="space-y-1 text-sm">
              <div className="flex items-center gap-2">
                {job.sandbox_results && job.sandbox_results.coverage >= gate.test_coverage_min ? (
                  <CheckCircle className="w-4 h-4 text-green-400" />
                ) : (
                  <XCircle className="w-4 h-4 text-red-400" />
                )}
                <span className="text-dark-400">Coverage ≥ {gate.test_coverage_min}%</span>
              </div>
              <div className="flex items-center gap-2">
                {job.sandbox_results && job.sandbox_results.performance_impact <= gate.performance_degradation_max ? (
                  <CheckCircle className="w-4 h-4 text-green-400" />
                ) : (
                  <XCircle className="w-4 h-4 text-red-400" />
                )}
                <span className="text-dark-400">Performance impact ≤ {gate.performance_degradation_max}%</span>
              </div>
              {gate.human_approval && (
                <div className="flex items-center gap-2">
                  <Clock className="w-4 h-4 text-yellow-400" />
                  <span className="text-dark-400">Awaiting human approval</span>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="flex justify-end gap-2 p-4 border-t border-dark-700">
          {needsApproval && (
            <>
              <button
                onClick={onRollback}
                className="px-4 py-2 bg-red-600 hover:bg-red-500 text-white text-sm rounded"
              >
                Reject
              </button>
              <button
                onClick={onApprove}
                disabled={!canPromote}
                className="px-4 py-2 bg-green-600 hover:bg-green-500 disabled:bg-dark-700 text-white text-sm rounded"
              >
                Approve & Promote
              </button>
            </>
          )}
          {job.status === "promoted" && (
            <button
              onClick={onRollback}
              className="px-4 py-2 bg-red-600 hover:bg-red-500 text-white text-sm rounded flex items-center gap-1.5"
            >
              <RotateCcw className="w-4 h-4" />
              Rollback
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
