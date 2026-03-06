import { useState, useEffect } from "react";
import { Camera, Users, CheckSquare, Clock, AlertTriangle, RefreshCw, Plus, UserCheck, UserX } from "lucide-react";
import { apiGet, apiPost } from "../lib/api-config";

export function CompanyCameraPanel() {
    const [summary, setSummary] = useState<any>(null);
    const [employees, setEmployees] = useState<any>({});
    const [tasks, setTasks] = useState<any>({});
    const [loading, setLoading] = useState(false);
    const [showAddForm, setShowAddForm] = useState(false);
    const [newEmp, setNewEmp] = useState({ id: "", name: "", nameAr: "", dept: "", role: "" });
    const [newTask, setNewTask] = useState({ title: "", dept: "", priority: 3 });

    const fetchAll = async () => {
        setLoading(true);
        try {
            const [sum, emps, tsks] = await Promise.all([
                apiGet("/api/company/summary").catch(() => null),
                apiGet("/api/company/employees").catch(() => ({})),
                apiGet("/api/company/tasks").catch(() => ({})),
            ]);
            setSummary(sum);
            setEmployees(emps || {});
            setTasks(tsks || {});
        } finally { setLoading(false); }
    };

    useEffect(() => { fetchAll(); }, []);

    const checkIn = async (empId: string) => {
        await apiPost("/api/company/check-in", { employee_id: empId });
        await fetchAll();
    };
    const checkOut = async (empId: string) => {
        await apiPost("/api/company/check-out", { employee_id: empId });
        await fetchAll();
    };
    const addEmployee = async () => {
        if (!newEmp.id || !newEmp.nameAr) return;
        await apiPost("/api/company/employees", {
            employee_id: newEmp.id, name: newEmp.name || newEmp.nameAr,
            name_ar: newEmp.nameAr, department: newEmp.dept, role: newEmp.role,
        });
        setNewEmp({ id: "", name: "", nameAr: "", dept: "", role: "" });
        setShowAddForm(false);
        await fetchAll();
    };
    const createTask = async () => {
        if (!newTask.title) return;
        await apiPost("/api/company/tasks", {
            title: newTask.title, department: newTask.dept, priority: newTask.priority,
        });
        setNewTask({ title: "", dept: "", priority: 3 });
        await fetchAll();
    };
    const completeTask = async (taskId: string) => {
        await apiPost(`/api/company/tasks/${taskId}/complete`, {});
        await fetchAll();
    };

    const empList = Object.values(employees) as any[];
    const taskList = Object.values(tasks) as any[];
    const presentEmps = empList.filter((e: any) => e.status === "حاضر" || e.status === "متأخر");
    const pendingTasks = taskList.filter((t: any) => t.status !== "مكتمل");

    return (
        <div className="p-3">
            <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                    <Camera className="w-4 h-4 text-blue-400" />
                    <span className="text-sm font-bold text-dark-200">إدارة الشركة 🏢</span>
                </div>
                <button onClick={fetchAll} disabled={loading} className="p-1 hover:bg-dark-800 rounded">
                    <RefreshCw className={`w-3 h-3 text-dark-400 ${loading ? "animate-spin" : ""}`} />
                </button>
            </div>

            {/* Summary cards */}
            {summary && (
                <div className="grid grid-cols-4 gap-1 mb-3">
                    <div className="bg-dark-800 rounded p-1.5 text-center">
                        <div className="text-[10px] text-dark-400">موظفين</div>
                        <div className="text-sm font-bold text-blue-400">{summary.employees}</div>
                    </div>
                    <div className="bg-dark-800 rounded p-1.5 text-center">
                        <div className="text-[10px] text-dark-400">حاضر</div>
                        <div className="text-sm font-bold text-green-400">{summary.present}</div>
                    </div>
                    <div className="bg-dark-800 rounded p-1.5 text-center">
                        <div className="text-[10px] text-dark-400">كاميرات</div>
                        <div className="text-sm font-bold text-purple-400">{summary.cameras}</div>
                    </div>
                    <div className="bg-dark-800 rounded p-1.5 text-center">
                        <div className="text-[10px] text-dark-400">مهام</div>
                        <div className="text-sm font-bold text-yellow-400">{summary.total_tasks}</div>
                    </div>
                </div>
            )}

            {/* Employees */}
            <div className="mb-3">
                <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-1">
                        <Users className="w-3 h-3 text-blue-400" />
                        <span className="text-xs font-medium text-dark-300">الموظفين ({empList.length})</span>
                    </div>
                    <button onClick={() => setShowAddForm(!showAddForm)} className="p-0.5 hover:bg-dark-700 rounded">
                        <Plus className="w-3 h-3 text-dark-400" />
                    </button>
                </div>
                {showAddForm && (
                    <div className="bg-dark-800 rounded p-2 mb-1 space-y-1">
                        <input value={newEmp.id} onChange={e => setNewEmp({ ...newEmp, id: e.target.value })}
                            placeholder="ID" className="w-full bg-dark-700 rounded px-2 py-0.5 text-[10px] text-dark-200" />
                        <input value={newEmp.nameAr} onChange={e => setNewEmp({ ...newEmp, nameAr: e.target.value })}
                            placeholder="الاسم" className="w-full bg-dark-700 rounded px-2 py-0.5 text-[10px] text-dark-200" />
                        <input value={newEmp.dept} onChange={e => setNewEmp({ ...newEmp, dept: e.target.value })}
                            placeholder="القسم" className="w-full bg-dark-700 rounded px-2 py-0.5 text-[10px] text-dark-200" />
                        <button onClick={addEmployee}
                            className="w-full py-0.5 bg-blue-700 hover:bg-blue-600 rounded text-[10px] text-white">
                            + إضافة
                        </button>
                    </div>
                )}
                <div className="space-y-0.5 max-h-[150px] overflow-auto">
                    {empList.map((emp: any) => (
                        <div key={emp.employee_id} className="flex items-center justify-between bg-dark-800 rounded px-2 py-1">
                            <div className="flex items-center gap-1">
                                <span className={`w-1.5 h-1.5 rounded-full ${emp.status === "حاضر" ? "bg-green-400" :
                                        emp.status === "متأخر" ? "bg-yellow-400" : "bg-dark-500"
                                    }`} />
                                <span className="text-[10px] text-dark-200">{emp.name_ar}</span>
                                <span className="text-[9px] text-dark-500">{emp.department}</span>
                            </div>
                            <div className="flex gap-0.5">
                                <button onClick={() => checkIn(emp.employee_id)} title="حضور"
                                    className="p-0.5 hover:bg-green-800 rounded">
                                    <UserCheck className="w-2.5 h-2.5 text-green-500" />
                                </button>
                                <button onClick={() => checkOut(emp.employee_id)} title="انصراف"
                                    className="p-0.5 hover:bg-red-800 rounded">
                                    <UserX className="w-2.5 h-2.5 text-red-500" />
                                </button>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Tasks */}
            <div className="mb-3">
                <div className="flex items-center gap-1 mb-1">
                    <CheckSquare className="w-3 h-3 text-yellow-400" />
                    <span className="text-xs font-medium text-dark-300">المهام ({pendingTasks.length} قيد التنفيذ)</span>
                </div>
                <div className="bg-dark-800 rounded p-2 mb-1 space-y-1">
                    <input value={newTask.title} onChange={e => setNewTask({ ...newTask, title: e.target.value })}
                        placeholder="مهمة جديدة..." className="w-full bg-dark-700 rounded px-2 py-0.5 text-[10px] text-dark-200" />
                    <div className="flex gap-1">
                        <input value={newTask.dept} onChange={e => setNewTask({ ...newTask, dept: e.target.value })}
                            placeholder="القسم" className="flex-1 bg-dark-700 rounded px-2 py-0.5 text-[10px] text-dark-200" />
                        <button onClick={createTask}
                            className="px-2 py-0.5 bg-yellow-700 hover:bg-yellow-600 rounded text-[10px] text-white">
                            + مهمة
                        </button>
                    </div>
                </div>
                <div className="space-y-0.5 max-h-[120px] overflow-auto">
                    {taskList.map((task: any) => (
                        <div key={task.task_id} className="flex items-center justify-between bg-dark-800 rounded px-2 py-1">
                            <div className="flex-1 min-w-0">
                                <span className={`text-[10px] ${task.status === "مكتمل" ? "text-dark-500 line-through" : "text-dark-200"}`}>
                                    {task.title}
                                </span>
                                {task.assigned_to && (
                                    <span className="text-[9px] text-dark-500 ml-1">
                                        → {employees[task.assigned_to]?.name_ar || task.assigned_to}
                                    </span>
                                )}
                            </div>
                            {task.status !== "مكتمل" && (
                                <button onClick={() => completeTask(task.task_id)}
                                    className="p-0.5 hover:bg-green-800 rounded ml-1">
                                    <CheckSquare className="w-2.5 h-2.5 text-green-500" />
                                </button>
                            )}
                        </div>
                    ))}
                </div>
            </div>

            {/* Daily report */}
            {summary?.today && (
                <div>
                    <div className="flex items-center gap-1 mb-1">
                        <Clock className="w-3 h-3 text-cyan-400" />
                        <span className="text-xs font-medium text-dark-300">تقرير اليوم</span>
                    </div>
                    <div className="bg-dark-800 rounded p-2">
                        <div className="grid grid-cols-3 gap-1 text-[10px] text-dark-400">
                            <span>ساعات: <span className="text-cyan-400 font-bold">{summary.today.total_hours}</span></span>
                            <span>غياب: <span className="text-red-400 font-bold">{summary.today.absent}</span></span>
                            <span>تأخير: <span className="text-yellow-400 font-bold">{summary.today.late}</span></span>
                        </div>
                        {summary.today.camera_alerts > 0 && (
                            <div className="flex items-center gap-1 mt-1">
                                <AlertTriangle className="w-3 h-3 text-red-400" />
                                <span className="text-[10px] text-red-400 font-bold">
                                    {summary.today.camera_alerts} تنبيه سلامة!
                                </span>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
