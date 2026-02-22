/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'bi-primary': '#1a365d',
        'bi-secondary': '#2c5282',
        'bi-accent': '#4299e1',
        'bi-gold': '#d69e2e',
        'bi-dark': '#0d1b2a',
        'bi-light': '#e2e8f0',
        'alert-green': '#48bb78',
        'alert-yellow': '#ecc94b',
        'alert-orange': '#ed8936',
        'alert-red': '#f56565',
        'alert-black': '#1a202c',
      },
      fontFamily: {
        'arabic': ['Noto Sans Arabic', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
