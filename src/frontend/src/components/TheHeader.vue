<template>
    <header class="text-center mb-8">
      <div class="flex justify-end mb-4">
        <button 
          @click="$emit('toggle-theme')" 
          class="p-2 rounded-full bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
          aria-label="Toggle theme"
        >
          <svg v-if="isDark" xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
          </svg>
          <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
          </svg>
        </button>
      </div>
      <h1 class="text-3xl md:text-4xl font-bold text-primary-700 dark:text-primary-400 mb-2">
        AI Text Detector
      </h1>
      <p class="text-gray-600 dark:text-gray-400">
        Determine if text is AI-generated or human-written
      </p>
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