<template>
    <div class="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 transition-colors duration-300">
      <h2 class="text-xl font-bold text-center mb-4 text-gray-800 dark:text-gray-200">
        Analysis Result
      </h2>
      
      <div class="space-y-6">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div class="flex items-center gap-3">
            <div 
              class="flex items-center justify-center w-12 h-12 rounded-full"
              :class="result.prediction === 'AI' ? 'bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400' : 'bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400'"
            >
              <svg v-if="result.prediction === 'AI'" xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
              <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
            </div>
            <div>
              <div class="text-sm text-gray-500 dark:text-gray-400">Prediction</div>
              <div 
                class="text-xl font-bold"
                :class="result.prediction === 'AI' ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'"
              >
                {{ result.prediction }}
              </div>
            </div>
          </div>
          
          <div>
            <div class="text-sm text-gray-500 dark:text-gray-400">Confidence</div>
            <div class="text-xl font-bold text-gray-800 dark:text-gray-200">
              {{ formattedConfidence }}%
            </div>
          </div>
        </div>
        
        <div class="space-y-2">
          <div class="flex justify-between text-sm text-gray-500 dark:text-gray-400">
            <span>Human</span>
            <span>AI</span>
          </div>
          
          <div class="h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
            <div 
              class="h-full transition-all duration-500 ease-out"
              :class="result.prediction === 'AI' ? 'bg-red-500' : 'bg-green-500'"
              :style="{ width: `${result.prediction === 'AI' ? result.confidence * 100 : (1 - result.confidence) * 100}%`, marginLeft: `${result.prediction === 'AI' ? 0 : result.confidence * 100}%` }"
            ></div>
          </div>
          
          <div class="flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
        </div>
      </div>
    </div>
  </template>
  
  <script>
  import { computed } from 'vue'
  
  export default {
    props: {
      result: {
        type: Object,
        required: true
      }
    },
    
    setup(props) {
      const formattedConfidence = computed(() => {
        const confidence = props.result.prediction === 'AI' ? 
          props.result.confidence : 
          1 - props.result.confidence
        
        return (confidence * 100).toFixed(1)
      })
      
      return {
        formattedConfidence
      }
    }
  }
  </script>