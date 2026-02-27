interface LoadingSpinnerProps {
  message?: string;
  size?: 'sm' | 'md' | 'lg';
}

const sizeClasses = {
  sm: 'w-8 h-8 border-2',
  md: 'w-12 h-12 border-3',
  lg: 'w-16 h-16 border-4',
};

export function LoadingSpinner({ message = 'جاري التحميل...', size = 'md' }: LoadingSpinnerProps) {
  return (
    <div className="flex flex-col items-center justify-center">
      <div className={`${sizeClasses[size]} border-bi-accent border-t-transparent rounded-full animate-spin mb-2`}></div>
      {message && <p className="text-gray-400 text-sm">{message}</p>}
    </div>
  );
}

export default LoadingSpinner;
