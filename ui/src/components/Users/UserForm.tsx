import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { api } from '@/lib/api';
import { User, Lock, Mail, UserCircle, Shield } from 'lucide-react';

interface UserFormData {
  id?: string;
  username: string;
  email: string;
  full_name: string;
  password?: string;
  role: string;
  is_active: boolean;
}

interface UserFormProps {
  user?: UserFormData | null;
  onSubmit?: (data: UserFormData) => void;
  onCancel?: () => void;
  isEditing?: boolean;
}

const ROLES = [
  { value: 'admin', label: 'مدير نظام' },
  { value: 'manager', label: 'مدير' },
  { value: 'accountant', label: 'محاسب' },
  { value: 'employee', label: 'موظف' },
];

export const UserForm: React.FC<UserFormProps> = ({ 
  user,
  onSubmit,
  onCancel,
  isEditing = false
}) => {
  const [formData, setFormData] = useState<UserFormData>({
    username: '',
    email: '',
    full_name: '',
    password: '',
    role: 'employee',
    is_active: true,
    ...user
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (user) {
      setFormData(prev => ({ ...prev, ...user }));
    }
  }, [user]);

  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};
    
    if (!formData.username.trim()) {
      newErrors.username = 'اسم المستخدم مطلوب';
    }
    if (!formData.email.trim()) {
      newErrors.email = 'البريد الإلكتروني مطلوب';
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) {
      newErrors.email = 'البريد الإلكتروني غير صالح';
    }
    if (!formData.full_name.trim()) {
      newErrors.full_name = 'الاسم الكامل مطلوب';
    }
    if (!isEditing && !formData.password) {
      newErrors.password = 'كلمة المرور مطلوبة';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) return;

    setLoading(true);
    try {
      if (isEditing && formData.id) {
        await api.put(`/users/${formData.id}`, formData);
      } else {
        await api.post('/users', formData);
      }
      onSubmit?.(formData);
    } catch (error) {
      console.error('Failed to save user:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (field: keyof UserFormData, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (errors[field]) {
      setErrors(prev => ({ ...prev, [field]: '' }));
    }
  };

  return (
    <Card className="w-full max-w-lg">
      <CardHeader>
        <div className="flex items-center gap-2">
          <UserCircle className="w-5 h-5 text-bi-accent" />
          <CardTitle>{isEditing ? 'تعديل مستخدم' : 'مستخدم جديد'}</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              الاسم الكامل
            </label>
            <div className="relative">
              <User className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <Input
                value={formData.full_name}
                onChange={(e) => handleChange('full_name', e.target.value)}
                className="pr-10"
                placeholder="محمد أحمد"
              />
            </div>
            {errors.full_name && (
              <p className="text-xs text-red-400 mt-1">{errors.full_name}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              اسم المستخدم
            </label>
            <div className="relative">
              <UserCircle className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <Input
                value={formData.username}
                onChange={(e) => handleChange('username', e.target.value)}
                className="pr-10"
                placeholder="mohammed.ahmed"
              />
            </div>
            {errors.username && (
              <p className="text-xs text-red-400 mt-1">{errors.username}</p>
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              البريد الإلكتروني
            </label>
            <div className="relative">
              <Mail className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <Input
                type="email"
                value={formData.email}
                onChange={(e) => handleChange('email', e.target.value)}
                className="pr-10"
                placeholder="mohammed@example.com"
              />
            </div>
            {errors.email && (
              <p className="text-xs text-red-400 mt-1">{errors.email}</p>
            )}
          </div>

          {!isEditing && (
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-1">
                كلمة المرور
              </label>
              <div className="relative">
                <Lock className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <Input
                  type="password"
                  value={formData.password}
                  onChange={(e) => handleChange('password', e.target.value)}
                  className="pr-10"
                  placeholder="••••••••"
                />
              </div>
              {errors.password && (
                <p className="text-xs text-red-400 mt-1">{errors.password}</p>
              )}
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-1">
              <Shield className="w-4 h-4 inline-block ml-1" />
              الدور
            </label>
            <select
              value={formData.role}
              onChange={(e) => handleChange('role', e.target.value)}
              className="input-field w-full"
            >
              {ROLES.map(role => (
                <option key={role.value} value={role.value}>
                  {role.label}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="is_active"
              checked={formData.is_active}
              onChange={(e) => handleChange('is_active', e.target.checked)}
              className="w-4 h-4 rounded border-gray-600 bg-gray-700 text-bi-accent"
            />
            <label htmlFor="is_active" className="text-sm text-gray-300">
              نشط
            </label>
          </div>

          <div className="flex gap-2 pt-4">
            <Button type="submit" disabled={loading} className="flex-1">
              {loading ? 'جاري الحفظ...' : (isEditing ? 'تحديث' : 'إنشاء')}
            </Button>
            <Button type="button" variant="secondary" onClick={onCancel}>
              إلغاء
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
};
