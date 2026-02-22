// أنواع النظام الهرمي
export type AlertLevel = 'GREEN' | 'YELLOW' | 'ORANGE' | 'RED' | 'BLACK';

export interface President {
  is_present: boolean;
  veto_power_active: boolean;
  current_meeting?: CouncilMeeting;
}

export interface CouncilMeeting {
  meeting_id: string;
  start_time: string;
  topic: string;
  participating_sages: string[];
  status: 'ongoing' | 'paused' | 'completed';
}

export interface WiseMan {
  id: string;
  name: string;
  role: string;
  specialty: string;
  is_active: boolean;
  current_task?: string;
}

export interface SystemStatus {
  president: {
    in_meeting: boolean;
    veto_power: boolean;
  };
  council: {
    is_meeting: boolean;
    wise_men_count: number;
    meeting_status: string;
    president_present: boolean;
  };
  scouts: {
    intel_buffer_size: number;
    high_priority_queue: number;
  };
  meta: {
    performance_score: number;
    quality_score: number;
    evolution_stage: number;
    learning_progress: number;
    status: string;
  };
  experts: {
    total: number;
    domains: string[];
  };
  execution: {
    active_forces: number;
    active_sprints: number;
    active_crises: number;
    quality_score: number;
  };
}

// أنواع ERP
export interface Invoice {
  id: string;
  customer_id: string;
  customer_name: string;
  amount: number;
  status: 'pending' | 'paid' | 'overdue';
  date: string;
  due_date: string;
}

export interface InventoryItem {
  id: string;
  name: string;
  sku: string;
  quantity: number;
  reorder_point: number;
  unit_price: number;
  category: string;
}

export interface Employee {
  id: string;
  name: string;
  position: string;
  department: string;
  salary: number;
  join_date: string;
  status: 'active' | 'inactive';
}

// أنواع Community
export interface Post {
  id: string;
  author: string;
  content: string;
  likes: number;
  comments: number;
  timestamp: string;
}

export interface Product {
  id: string;
  name: string;
  description: string;
  price: number;
  seller: string;
  category: string;
}

// أنواع التدريب
export interface TrainingShard {
  id: string;
  name: string;
  size: string;
  status: 'pending' | 'training' | 'completed';
  progress: number;
}

export interface TrainingStatus {
  is_training: boolean;
  current_shard: number;
  total_shards: number;
  progress: number;
  loss: number;
  epoch: number;
}
