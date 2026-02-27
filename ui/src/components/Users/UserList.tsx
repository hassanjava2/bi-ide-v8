import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { api } from '@/lib/api';
import { Users, Search, Edit, Trash2, UserPlus, CheckCircle, XCircle } from 'lucide-react';

interface User {
  id: string;
  username: string;
  email: string;
  full_name: string;
  role: string;
  is_active: boolean;
  permissions: string[];
  created_at: string;
}

interface UserListProps {
  users?: User[];
  onEdit?: (user: User) => void;
  onDelete?: (userId: string) => void;
  onCreate?: () => void;
}

export const UserList: React.FC<UserListProps> = ({ 
  users: initialUsers,
  onEdit,
  onDelete,
  onCreate
}) => {
  const [users, setUsers] = useState<User[]>(initialUsers || []);
  const [searchQuery, setSearchQuery] = useState('');
  const [loading, setLoading] = useState(!initialUsers);

  useEffect(() => {
    if (!initialUsers) {
      fetchUsers();
    }
  }, [initialUsers]);

  const fetchUsers = async () => {
    try {
      const response = await api.get('/users');
      setUsers(response.data.users || []);
    } catch (error) {
      console.error('Failed to fetch users:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleActive = async (userId: string, currentStatus: boolean) => {
    try {
      await api.post(`/users/${userId}/toggle-active`, { is_active: !currentStatus });
      setUsers(prev => prev.map(u => 
        u.id === userId ? { ...u, is_active: !currentStatus } : u
      ));
    } catch (error) {
      console.error('Failed to toggle user status:', error);
    }
  };

  const filteredUsers = users.filter(user => 
    user.username.toLowerCase().includes(searchQuery.toLowerCase()) ||
    user.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
    user.full_name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getRoleBadgeClass = (role: string) => {
    switch (role) {
      case 'admin': return 'bg-red-500/20 text-red-400';
      case 'manager': return 'bg-blue-500/20 text-blue-400';
      case 'accountant': return 'bg-green-500/20 text-green-400';
      case 'employee': return 'bg-gray-500/20 text-gray-400';
      default: return 'bg-gray-500/20 text-gray-400';
    }
  };

  const getRoleLabel = (role: string) => {
    switch (role) {
      case 'admin': return 'مدير نظام';
      case 'manager': return 'مدير';
      case 'accountant': return 'محاسب';
      case 'employee': return 'موظف';
      default: return role;
    }
  };

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>المستخدمين</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-center text-gray-400 py-4">جاري التحميل...</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Users className="w-5 h-5 text-bi-accent" />
            <CardTitle>المستخدمين</CardTitle>
          </div>
          <Button onClick={onCreate} size="sm">
            <UserPlus className="w-4 h-4 mr-1" />
            مستخدم جديد
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <div className="mb-4 relative">
          <Search className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <Input
            placeholder="بحث في المستخدمين..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pr-10"
          />
        </div>

        <div className="space-y-2 max-h-96 overflow-y-auto">
          {filteredUsers.map((user) => (
            <div
              key={user.id}
              className="flex items-center justify-between p-3 bg-white/5 rounded-lg hover:bg-white/10 transition-colors"
            >
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                  user.is_active ? 'bg-green-500/20' : 'bg-gray-500/20'
                }`}>
                  <span className="text-sm font-medium">
                    {user.full_name.charAt(0)}
                  </span>
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <p className="font-medium text-sm">{user.full_name}</p>
                    <span className={`text-xs px-2 py-0.5 rounded ${getRoleBadgeClass(user.role)}`}>
                      {getRoleLabel(user.role)}
                    </span>
                  </div>
                  <p className="text-xs text-gray-400">{user.email}</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleToggleActive(user.id, user.is_active)}
                  className="p-2 hover:bg-white/10 rounded"
                  title={user.is_active ? 'تعطيل' : 'تفعيل'}
                >
                  {user.is_active ? (
                    <CheckCircle className="w-4 h-4 text-green-400" />
                  ) : (
                    <XCircle className="w-4 h-4 text-gray-400" />
                  )}
                </button>
                <button
                  onClick={() => onEdit?.(user)}
                  className="p-2 hover:bg-white/10 rounded"
                  title="تعديل"
                >
                  <Edit className="w-4 h-4 text-blue-400" />
                </button>
                <button
                  onClick={() => onDelete?.(user.id)}
                  className="p-2 hover:bg-white/10 rounded"
                  title="حذف"
                >
                  <Trash2 className="w-4 h-4 text-red-400" />
                </button>
              </div>
            </div>
          ))}
          {filteredUsers.length === 0 && (
            <p className="text-center text-gray-400 py-4">لا يوجد مستخدمين</p>
          )}
        </div>

        <div className="mt-4 pt-4 border-t border-white/10">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">إجمالي المستخدمين: {users.length}</span>
            <span className="text-green-400">نشط: {users.filter(u => u.is_active).length}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
};
