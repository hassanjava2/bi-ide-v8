//! File Types for Desktop App

export interface FileNode {
  path: string;
  name: string;
  isDir: boolean;
  size: number;
  modifiedAt?: number;
  children?: FileNode[];
  isExpanded?: boolean;
}

export interface OpenFile {
  path: string;
  content: string;
  isModified: boolean;
  isDirty: boolean;
}

export interface FileInfo {
  path: string;
  name: string;
  is_dir: boolean;
  size: number;
  modified_at?: number;
  created_at?: number;
}
