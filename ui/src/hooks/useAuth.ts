import { useState, useEffect, useCallback } from 'react';
import { authApi } from '@/lib/api';

export interface User {
  id: string;
  username: string;
  email: string;
  first_name?: string;
  last_name?: string;
  role: string;
  is_active: boolean;
}

export interface AuthState {
  user: User | null;
  loading: boolean;
  error: string | null;
  isAuthenticated: boolean;
}

export const useAuth = () => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (token) {
      fetchUser();
    } else {
      setLoading(false);
    }
  }, []);

  const fetchUser = async () => {
    try {
      const data = await authApi.getMe();
      setUser(data);
    } catch (err) {
      localStorage.removeItem('access_token');
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const login = useCallback(async (username: string, password: string): Promise<boolean> => {
    setLoading(true);
    setError(null);
    try {
      const data = await authApi.login(username, password);
      localStorage.setItem('access_token', data.access_token);
      setUser(data.user);
      return true;
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Login failed');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const register = useCallback(async (userData: {
    username: string;
    email: string;
    password: string;
    first_name?: string;
    last_name?: string;
  }): Promise<boolean> => {
    setLoading(true);
    setError(null);
    try {
      await authApi.register(userData);
      return true;
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Registration failed');
      return false;
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('access_token');
    setUser(null);
    setError(null);
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    user,
    loading,
    error,
    login,
    logout,
    register,
    clearError,
    isAuthenticated: !!user,
  };
};

export default useAuth;
