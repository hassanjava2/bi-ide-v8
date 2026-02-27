import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { api } from '@/lib/api';
import { Shield, Plus, Trash2, Edit, Check, X } from 'lucide-react';

interface Permission {
  id: string;
  name: string;
  description: string;
  module: string;
}

interface Role {
  id: string;
  name: string;
  label: string;
  permissions: string[];
  user_count: number;
}

interface RoleManagerProps {
  roles?: Role[];
  permissions?: Permission[];
  onUpdate?: () => void;
}

const DEFAULT_PERMISSIONS: Permission[] = [
  { id: 'erp.view', name: 'عرض ERP', description: 'عرض لوحة تحكم ERP', module: 'erp' },
  { id: 'erp.invoices.manage', name: 'إدارة الفواتير', description: 'إنشاء وتعديل الفواتير', module: 'erp' },
  { id: 'erp.inventory.manage', name: 'إدارة المخزون', description: 'إدارة المنتجات والمخزون', module: 'erp' },
  { id: 'erp.accounts.manage', name: 'إدارة الحسابات', description: 'إدارة الحسابات المحاسبية', module: 'erp' },
  { id: 'users.view', name: 'عرض المستخدمين', description: 'عرض قائمة المستخدمين', module: 'users' },
  { id: 'users.manage', name: 'إدارة المستخدمين', description: 'إنشاء وتعديل وحذف المستخدمين', module: 'users' },
  { id: 'roles.manage', name: 'إدارة الأدوار', description: 'إدارة الأدوار والصلاحيات', module: 'users' },
  { id: 'council.access', name: 'الوصول للمجلس', description: 'الوصول إلى مجلس الحكماء', module: 'council' },
  { id: 'ide.access', name: 'الوصول لل IDE', description: 'الوصول إلى بيئة التطوير', module: 'ide' },
  { id: 'settings.manage', name: 'إدارة الإعدادات', description: 'تعديل إعدادات النظام', module: 'settings' },
];

export const RoleManager: React.FC<RoleManagerProps> = ({ 
  roles: initialRoles,
  permissions: initialPermissions,
  onUpdate
}) => {
  const [roles, setRoles] = useState<Role[]>(initialRoles || []);
  const [permissions] = useState<Permission[]>(initialPermissions || DEFAULT_PERMISSIONS);
  const [editingRole, setEditingRole] = useState<string | null>(null);
  const [newRoleName, setNewRoleName] = useState('');
  const [newRoleLabel, setNewRoleLabel] = useState('');
  const [selectedPermissions, setSelectedPermissions] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!initialRoles) {
      fetchRoles();
    }
  }, [initialRoles]);

  const fetchRoles = async () => {
    try {
      const response = await api.get('/roles');
      setRoles(response.data.roles || []);
    } catch (error) {
      console.error('Failed to fetch roles:', error);
    }
  };

  const handleCreateRole = async () => {
    if (!newRoleName || !newRoleLabel) return;

    try {
      await api.post('/roles', {
        name: newRoleName,
        label: newRoleLabel,
        permissions: Array.from(selectedPermissions)
      });
      setNewRoleName('');
      setNewRoleLabel('');
      setSelectedPermissions(new Set());
      fetchRoles();
      onUpdate?.();
    } catch (error) {
      console.error('Failed to create role:', error);
    }
  };

  const handleUpdateRole = async (roleId: string) => {
    try {
      await api.put(`/roles/${roleId}`, {
        permissions: Array.from(selectedPermissions)
      });
      setEditingRole(null);
      fetchRoles();
      onUpdate?.();
    } catch (error) {
      console.error('Failed to update role:', error);
    }
  };

  const handleDeleteRole = async (roleId: string) => {
    if (!confirm('هل أنت متأكد من حذف هذا الدور؟')) return;

    try {
      await api.delete(`/roles/${roleId}`);
      fetchRoles();
      onUpdate?.();
    } catch (error) {
      console.error('Failed to delete role:', error);
    }
  };

  const startEditing = (role: Role) => {
    setEditingRole(role.id);
    setSelectedPermissions(new Set(role.permissions));
  };

  const togglePermission = (permissionId: string) => {
    setSelectedPermissions(prev => {
      const newSet = new Set(prev);
      if (newSet.has(permissionId)) {
        newSet.delete(permissionId);
      } else {
        newSet.add(permissionId);
      }
      return newSet;
    });
  };

  const groupedPermissions = permissions.reduce((acc, perm) => {
    if (!acc[perm.module]) acc[perm.module] = [];
    acc[perm.module].push(perm);
    return acc;
  }, {} as Record<string, Permission[]>);

  const moduleLabels: Record<string, string> = {
    erp: 'نظام ERP',
    users: 'المستخدمين',
    council: 'المجلس',
    ide: 'بيئة التطوير',
    settings: 'الإعدادات',
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Shield className="w-5 h-5 text-bi-accent" />
          <CardTitle>إدارة الأدوار والصلاحيات</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <div className="mb-6 p-4 bg-white/5 rounded-lg">
          <h4 className="text-sm font-medium mb-3">دور جديد</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-3">
            <Input
              placeholder="معرف الدور (مثال: manager)"
              value={newRoleName}
              onChange={(e) => setNewRoleName(e.target.value)}
            />
            <Input
              placeholder="اسم الدور (مثال: مدير)"
              value={newRoleLabel}
              onChange={(e) => setNewRoleLabel(e.target.value)}
            />
          </div>
          <Button onClick={handleCreateRole} size="sm" disabled={!newRoleName || !newRoleLabel}>
            <Plus className="w-4 h-4 mr-1" />
            إضافة دور
          </Button>
        </div>

        <div className="space-y-4">
          {roles.map((role) => (
            <div key={role.id} className="border border-white/10 rounded-lg overflow-hidden">
              <div className="flex items-center justify-between p-3 bg-white/5">
                <div>
                  <h4 className="font-medium">{role.label}</h4>
                  <p className="text-xs text-gray-400">
                    {role.name} | {role.user_count} مستخدم | {role.permissions.length} صلاحية
                  </p>
                </div>
                <div className="flex items-center gap-1">
                  {editingRole === role.id ? (
                    <>
                      <button
                        onClick={() => handleUpdateRole(role.id)}
                        className="p-2 hover:bg-green-500/20 rounded text-green-400"
                      >
                        <Check className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => setEditingRole(null)}
                        className="p-2 hover:bg-red-500/20 rounded text-red-400"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </>
                  ) : (
                    <>
                      <button
                        onClick={() => startEditing(role)}
                        className="p-2 hover:bg-white/10 rounded"
                      >
                        <Edit className="w-4 h-4 text-blue-400" />
                      </button>
                      <button
                        onClick={() => handleDeleteRole(role.id)}
                        className="p-2 hover:bg-white/10 rounded"
                      >
                        <Trash2 className="w-4 h-4 text-red-400" />
                      </button>
                    </>
                  )}
                </div>
              </div>

              {editingRole === role.id && (
                <div className="p-4 border-t border-white/10">
                  <h5 className="text-sm font-medium mb-3">الصلاحيات</h5>
                  <div className="space-y-4">
                    {Object.entries(groupedPermissions).map(([module, perms]) => (
                      <div key={module}>
                        <h6 className="text-xs font-medium text-gray-400 mb-2">
                          {moduleLabels[module] || module}
                        </h6>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                          {perms.map(perm => (
                            <label
                              key={perm.id}
                              className="flex items-center gap-2 p-2 bg-white/5 rounded cursor-pointer hover:bg-white/10"
                            >
                              <input
                                type="checkbox"
                                checked={selectedPermissions.has(perm.id)}
                                onChange={() => togglePermission(perm.id)}
                                className="w-4 h-4 rounded border-gray-600"
                              />
                              <div>
                                <p className="text-sm">{perm.name}</p>
                                <p className="text-xs text-gray-400">{perm.description}</p>
                              </div>
                            </label>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {editingRole !== role.id && role.permissions.length > 0 && (
                <div className="px-3 py-2 border-t border-white/10">
                  <div className="flex flex-wrap gap-1">
                    {role.permissions.slice(0, 5).map(permId => {
                      const perm = permissions.find(p => p.id === permId);
                      return perm ? (
                        <span
                          key={permId}
                          className="text-xs px-2 py-0.5 bg-bi-accent/20 text-bi-accent rounded"
                        >
                          {perm.name}
                        </span>
                      ) : null;
                    })}
                    {role.permissions.length > 5 && (
                      <span className="text-xs px-2 py-0.5 bg-gray-500/20 text-gray-400 rounded">
                        +{role.permissions.length - 5}
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          ))}
          {roles.length === 0 && (
            <p className="text-center text-gray-400 py-4">لا توجد أدوار معرفة</p>
          )}
        </div>
      </CardContent>
    </Card>
  );
};
