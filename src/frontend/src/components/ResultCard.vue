<template>
    <div class="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-2xl shadow-xl p-8 transition-all duration-300 transform hover:shadow-2xl relative overflow-hidden h-full">
      <div class="absolute -right-16 -top-16 w-32 h-32 rounded-full opacity-10"
           :class="result.prediction === 'AI' ? 'bg-red-500 dark:bg-red-400' : 'bg-green-500 dark:bg-green-400'"></div>
      <div class="absolute -left-16 -bottom-16 w-32 h-32 rounded-full opacity-10"
           :class="result.prediction === 'AI' ? 'bg-red-500 dark:bg-red-400' : 'bg-green-500 dark:bg-green-400'"></div>
      
      <h2 class="text-2xl font-bold text-center mb-6 text-gray-800 dark:text-gray-200 relative">
        Analysis Result
        <div class="h-1 w-20 mx-auto mt-2 rounded-full"
             :class="result.prediction === 'AI' ? 'bg-red-500 dark:bg-red-400' : 'bg-green-500 dark:bg-green-400'"></div>
      </h2>
      
      <div class="space-y-8">
        <div class="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div class="flex items-center gap-4">
            <div 
              class="flex items-center justify-center w-16 h-16 rounded-2xl shadow-lg transform transition-transform duration-500 hover:scale-110"
              :class="result.prediction === 'AI' ? 
                'bg-gradient-to-br from-red-500 to-red-600 dark:from-red-600 dark:to-red-700 text-white' : 
                'bg-gradient-to-br from-green-500 to-green-600 dark:from-green-600 dark:to-green-700 text-white'"
            >
              <svg v-if="result.prediction === 'AI'" xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
              </svg>
              <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
              </svg>
            </div>
            <div>
              <div class="text-sm text-gray-500 dark:text-gray-400">Prediction</div>
              <div 
                class="text-3xl font-bold tracking-tight"
                :class="result.prediction === 'AI' ? 'text-red-600 dark:text-red-400' : 'text-green-600 dark:text-green-400'"
              >
                {{ result.prediction }}
                <span class="text-sm font-normal text-gray-500 dark:text-gray-400">generated</span>
              </div>
            </div>
          </div>
          
          <div class="bg-gray-100 dark:bg-gray-700/50 rounded-xl p-4 shadow-inner">
            <div class="text-sm text-gray-500 dark:text-gray-400">Confidence</div>
            <div class="text-3xl font-bold text-gray-800 dark:text-gray-200 flex items-baseline">
              {{ formattedConfidence }}
              <span class="text-xl ml-0.5">%</span>
              <span class="ml-2 text-xs px-2 py-0.5 rounded-full"
                    :class="confidenceLevel.color">
                {{ confidenceLevel.label }}
              </span>
            </div>
          </div>
        </div>
        
        <div class="space-y-3">
          <div class="flex justify-between text-sm font-medium">
            <span class="text-green-600 dark:text-green-400">Human</span>
            <span class="text-red-600 dark:text-red-400">AI</span>
          </div>
          
          <div class="h-4 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden shadow-inner">
            <div 
              class="h-full transition-all duration-1000 ease-out relative"
              :class="result.prediction === 'AI' ? 'bg-gradient-to-r from-red-400 to-red-600 dark:from-red-500 dark:to-red-700' : 'bg-gradient-to-r from-green-400 to-green-600 dark:from-green-500 dark:to-green-700'"
              :style="{ width: `${result.prediction === 'AI' ? result.confidence * 100 : (1 - result.confidence) * 100}%`, marginLeft: `${result.prediction === 'AI' ? 0 : result.confidence * 100}%` }"
            >
              <div class="absolute inset-0 opacity-30">
                <div class="w-full h-full bg-stripes"></div>
              </div>
            </div>
          </div>
          
          <div class="flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>0%</span>
            <span>50%</span>
            <span>100%</span>
          </div>
        </div>
        
        <div class="p-4 rounded-xl bg-gray-100 dark:bg-gray-700/50 text-sm text-gray-600 dark:text-gray-300 leading-relaxed">
          <p v-if="result.prediction === 'AI'" class="mb-2">
            <span class="font-semibold">Analysis:</span> This text shows characteristics commonly found in AI-generated content, including:
          </p>
          <p v-else class="mb-2">
            <span class="font-semibold">Analysis:</span> This text shows characteristics commonly found in human-written content, including:
          </p>
          <ul class="list-disc list-inside space-y-1 ml-2">
            <li v-if="result.prediction === 'AI'">Consistent patterning and predictable language structures</li>
            <li v-if="result.prediction === 'AI'">Limited stylistic variations and repetitive phrasing</li>
            <li v-if="result.prediction === 'AI'">Overly formal or generic tone throughout the text</li>
            <li v-if="result.prediction === 'Human'">Natural language variations and inconsistencies</li>
            <li v-if="result.prediction === 'Human'">Unique stylistic elements and personal voice</li>
            <li v-if="result.prediction === 'Human'">Contextually appropriate informal elements</li>
          </ul>
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
      
      const confidenceLevel = computed(() => {
        const confidence = parseFloat(formattedConfidence.value)
        
        if (confidence >= 90) {
          return {
            label: 'Very High',
            color: props.result.prediction === 'AI' ? 
              'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300' : 
              'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
          }
        } else if (confidence >= 75) {
          return {
            label: 'High',
            color: props.result.prediction === 'AI' ? 
              'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300' : 
              'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
          }
        } else if (confidence >= 60) {
          return {
            label: 'Moderate',
            color: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
          }
        } else {
          return {
            label: 'Low',
            color: 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300'
          }
        }
      })
      
      return {
        formattedConfidence,
        confidenceLevel
      }
    }
  }
  </script>
  
  <style>
  .bg-stripes {
    background-image: linear-gradient(
      45deg,
      rgba(255, 255, 255, 0.15) 25%,
      transparent 25%,
      transparent 50%,
      rgba(255, 255, 255, 0.15) 50%,
      rgba(255, 255, 255, 0.15) 75%,
      transparent 75%,
      transparent
    );
    background-size: 1rem 1rem;
    animation: stripe-animation 1s linear infinite;
  }
  
  @keyframes stripe-animation {
    from {
      background-position: 1rem 0;
    }
    to {
      background-position: 0 0;
    }
  }
  </style>