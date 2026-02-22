interface LoadingSpinnerProps {
  message?: string
}

function LoadingSpinner({ message = 'جاري التحميل...' }: LoadingSpinnerProps) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-bi-dark">
      <div className="text-center">
        <div className="w-16 h-16 border-4 border-bi-accent border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-gray-400">{message}</p>
      </div>
    </div>
  )
}

export default LoadingSpinner
