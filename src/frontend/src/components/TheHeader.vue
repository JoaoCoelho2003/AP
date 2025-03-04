<template>
    <header class="w-full bg-white/50 dark:bg-gray-900/50 backdrop-blur-sm shadow-sm">
      <div class="container mx-auto px-4 py-6">
        <div class="flex justify-between items-center">
          <div class="text-left">
            <h1 class="text-3xl md:text-4xl lg:text-5xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-primary-600 to-primary-400 dark:from-primary-400 dark:to-primary-300">
              AI Text Detector
            </h1>
            <p class="text-gray-600 dark:text-gray-400 text-sm md:text-base max-w-2xl leading-relaxed">
              Determine if text is AI-generated or human-written with advanced machine learning
            </p>
            <div class="mt-3 flex flex-wrap gap-2">
              <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-800 dark:bg-primary-900 dark:text-primary-300">
                <span class="w-2 h-2 mr-1 rounded-full bg-primary-500 animate-pulse"></span>
                Real-time
              </span>
              <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300">
                ML-powered
              </span>
              <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300">
                High accuracy
              </span>
            </div>
          </div>
          
          <button 
            @click="$emit('toggle-theme')" 
            class="p-3 rounded-full bg-gray-200/80 dark:bg-gray-700/80 backdrop-blur-sm text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600 transition-all duration-300 shadow-md hover:shadow-lg transform hover:scale-105 active:scale-95"
            aria-label="Toggle theme"
          >
            <div class="relative w-6 h-6">
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                class="h-6 w-6 absolute transition-transform duration-500 ease-spring" 
                :class="isDark ? 'rotate-0 opacity-100' : '-rotate-90 opacity-0'"
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
              </svg>
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                class="h-6 w-6 absolute transition-transform duration-500 ease-spring" 
                :class="isDark ? 'rotate-90 opacity-0' : 'rotate-0 opacity-100'"
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
              </svg>
            </div>
          </button>
        </div>
      </div>
    </header>
  </template>
    
  <script>
  export default {
    props: {
      isDark: {
        type: Boolean,
        required: true
      }
    },
    emits: ['toggle-theme']
  }
  </script>
    
  <style>
  .ease-spring {
    transition-timing-function: cubic-bezier(0.5, 0, 0.1, 1.4);
  }
  </style>