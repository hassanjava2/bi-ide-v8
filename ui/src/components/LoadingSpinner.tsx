//! Loading spinner component with variants
import { memo } from 'react'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'error'
  className?: string
  text?: string
}

const sizeClasses = {
  sm: 'w-4 h-4 border-2',
  md: 'w-6 h-6 border-2',
  lg: 'w-8 h-8 border-3',
  xl: 'w-12 h-12 border-4'
}

const variantClasses = {
  default: 'border-gray-600 border-t-transparent',
  primary: 'border-blue-600 border-t-transparent',
  success: 'border-green-600 border-t-transparent',
  warning: 'border-yellow-600 border-t-transparent',
  error: 'border-red-600 border-t-transparent'
}

export const LoadingSpinner = memo(function LoadingSpinner({
  size = 'md',
  variant = 'default',
  className = '',
  text
}: LoadingSpinnerProps) {
  return (
    <div className={`flex flex-col items-center justify-center gap-2 ${className}`}>
      <div
        className={`
          ${sizeClasses[size]}
          ${variantClasses[variant]}
          rounded-full animate-spin
        `}
      />
      {text && (
        <span className="text-sm text-gray-400">{text}</span>
      )}
    </div>
  )
})

// Full page loading
export function FullPageLoading({ text = 'Loading...' }: { text?: string }) {
  return (
    <div className="min-h-screen bg-gray-900 flex items-center justify-center">
      <div className="text-center">
        <LoadingSpinner size="xl" variant="primary" className="mb-4" />
        <p className="text-gray-400">{text}</p>
      </div>
    </div>
  )
}

// Skeleton loader
interface SkeletonProps {
  width?: string
  height?: string
  className?: string
  circle?: boolean
}

export function Skeleton({ width, height, className = '', circle = false }: SkeletonProps) {
  return (
    <div
      className={`
        bg-gray-800 animate-pulse
        ${circle ? 'rounded-full' : 'rounded'}
        ${className}
      `}
      style={{ width, height }}
    />
  )
}

// Card skeleton
export function CardSkeleton() {
  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-3">
      <Skeleton width="60%" height="20px" />
      <Skeleton width="100%" height="12px" />
      <Skeleton width="80%" height="12px" />
    </div>
  )
}

// List skeleton
export function ListSkeleton({ count = 3 }: { count?: number }) {
  return (
    <div className="space-y-2">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="flex items-center gap-3 p-3 bg-gray-800 rounded">
          <Skeleton width="40px" height="40px" circle />
          <div className="flex-1 space-y-2">
            <Skeleton width="40%" height="14px" />
            <Skeleton width="60%" height="10px" />
          </div>
        </div>
      ))}
    </div>
  )
}

// Table skeleton
export function TableSkeleton({ rows = 5, cols = 4 }: { rows?: number; cols?: number }) {
  return (
    <div className="w-full">
      {/* Header */}
      <div className="flex gap-2 mb-2 pb-2 border-b border-gray-700">
        {Array.from({ length: cols }).map((_, i) => (
          <Skeleton key={i} width={`${100 / cols}%`} height="16px" />
        ))}
      </div>
      {/* Rows */}
      {Array.from({ length: rows }).map((_, rowIndex) => (
        <div key={rowIndex} className="flex gap-2 mb-2">
          {Array.from({ length: cols }).map((_, colIndex) => (
            <Skeleton key={colIndex} width={`${100 / cols}%`} height="12px" />
          ))}
        </div>
      ))}
    </div>
  )
}

// Suspense fallback
export function SuspenseFallback() {
  return (
    <div className="p-8">
      <CardSkeleton />
    </div>
  )
}
