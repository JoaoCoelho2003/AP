<template>
  <div class="min-h-screen transition-all duration-500 relative overflow-hidden" 
       :class="{ 'dark': isDark, 'bg-gray-50 text-gray-900': !isDark, 'bg-gray-900 text-gray-100': isDark }">
    <div class="absolute inset-0 -z-10 overflow-hidden">
      <div class="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-primary-100/30 to-transparent dark:from-primary-900/20 dark:to-transparent"></div>
      <div class="absolute -top-[40%] -right-[30%] w-[80%] h-[80%] rounded-full bg-gradient-to-br from-primary-200/20 to-purple-300/20 dark:from-primary-800/10 dark:to-purple-900/10 blur-3xl"></div>
      <div class="absolute -bottom-[40%] -left-[30%] w-[80%] h-[80%] rounded-full bg-gradient-to-tr from-blue-200/20 to-primary-300/20 dark:from-blue-900/10 dark:to-primary-800/10 blur-3xl"></div>
      
      <div v-for="i in 8" :key="i" 
           class="absolute rounded-full opacity-30 dark:opacity-20 particle"
           :class="`particle-${i}`"
           :style="{
             width: `${10 + Math.random() * 20}px`,
             height: `${10 + Math.random() * 20}px`,
             background: isDark ? 
               `rgba(${100 + Math.random() * 155}, ${100 + Math.random() * 155}, ${200 + Math.random() * 55}, 0.3)` : 
               `rgba(${50 + Math.random() * 155}, ${100 + Math.random() * 155}, ${200 + Math.random() * 55}, 0.3)`,
             left: `${Math.random() * 100}%`,
             top: `${Math.random() * 100}%`,
             animationDuration: `${20 + Math.random() * 40}s`,
             animationDelay: `${Math.random() * 5}s`
           }">
      </div>
    </div>

    <div class="container mx-auto px-4 py-8 max-w-4xl relative">
      <TheHeader :isDark="isDark" @toggle-theme="toggleTheme" />
      
      <main class="my-8">
        <div class="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-2xl shadow-xl p-8 transition-all duration-300 transform hover:shadow-2xl">
          <div class="mb-8 relative group">
            <label for="text-input" class="block mb-2 font-medium text-gray-700 dark:text-gray-300 transition-all duration-300 group-focus-within:text-primary-600 dark:group-focus-within:text-primary-400">
              Enter text to analyze:
            </label>
            <div class="relative">
              <textarea 
                id="text-input" 
                v-model="text"
                placeholder="Paste or type text here..."
                rows="8"
                class="w-full px-5 py-4 rounded-xl border-2 border-gray-300 dark:border-gray-600 bg-white/90 dark:bg-gray-700/90 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all duration-300 resize-none"
                :disabled="isLoading"
                :class="{'shadow-inner': text.length > 0}"
              ></textarea>
              <div class="absolute bottom-3 right-3 text-xs text-gray-400 dark:text-gray-500" v-if="text.length > 0">
                {{ text.length }} characters
              </div>
            </div>
          </div>
          
          <div class="mb-8">
            <label class="block mb-2 font-medium text-gray-700 dark:text-gray-300">
              Select model:
            </label>
            <div class="relative">
              <button 
                type="button"
                @click="isDropdownOpen = !isDropdownOpen"
                class="w-full flex items-center justify-between px-5 py-4 bg-white/90 dark:bg-gray-700/90 border-2 border-gray-300 dark:border-gray-600 rounded-xl text-left focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500 transition-all duration-300 hover:border-primary-400 dark:hover:border-primary-500"
              >
                <span>{{ selectedModelName }}</span>
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  class="h-5 w-5 text-gray-500 dark:text-gray-400 transition-transform duration-300" 
                  :class="{ 'transform rotate-180': isDropdownOpen }"
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              
              <div 
                v-if="isDropdownOpen" 
                class="absolute z-10 w-full mt-2 bg-white/95 dark:bg-gray-800/95 backdrop-blur-sm border border-gray-200 dark:border-gray-700 rounded-xl shadow-xl overflow-hidden transform origin-top transition-all duration-200 animate-dropdown"
              >
                <div class="max-h-60 overflow-y-auto">
                  <button
                    v-for="model in models"
                    :key="model.id"
                    @click="selectModel(model.id)"
                    class="w-full px-5 py-4 text-left hover:bg-gray-100 dark:hover:bg-gray-700 transition-all duration-200"
                    :class="{ 'bg-primary-50 dark:bg-primary-900/20': selectedModel === model.id }"
                  >
                    <div class="flex items-center">
                      <div class="flex-1">
                        <div class="font-medium">{{ model.name }}</div>
                        <div v-if="model.description" class="text-sm text-gray-500 dark:text-gray-400">
                          {{ model.description }}
                        </div>
                      </div>
                      <svg 
                        v-if="selectedModel === model.id"
                        xmlns="http://www.w3.org/2000/svg" 
                        class="h-5 w-5 text-primary-600 dark:text-primary-400" 
                        fill="none" 
                        viewBox="0 0 24 24" 
                        stroke="currentColor"
                      >
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                  </button>
                </div>
              </div>
            </div>
          </div>
          
          <button 
            @click="predict" 
            :disabled="isLoading || !text.trim()"
            class="w-full px-6 py-4 rounded-xl font-medium transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 bg-gradient-to-r from-primary-600 to-primary-500 hover:from-primary-700 hover:to-primary-600 text-white focus:ring-primary-500 dark:from-primary-700 dark:to-primary-600 dark:hover:from-primary-800 dark:hover:to-primary-700 disabled:opacity-60 disabled:cursor-not-allowed disabled:hover:from-primary-600 disabled:hover:to-primary-500 dark:disabled:hover:from-primary-700 dark:disabled:hover:to-primary-600 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 active:translate-y-0"
          >
            <div class="flex items-center justify-center">
              <div v-if="isLoading" class="mr-3 relative w-5 h-5">
                <div class="loader-ring"></div>
              </div>
              <span class="text-lg">{{ isLoading ? 'Analyzing...' : 'Analyze Text' }}</span>
            </div>
          </button>
        </div>
        
        <transition name="fade-slide-up">
          <ResultCard v-if="result" :result="result" class="mt-8" />
        </transition>
        
        <transition name="fade-slide-up">
          <div v-if="error" class="mt-8 p-5 bg-red-100/80 dark:bg-red-900/30 backdrop-blur-sm text-red-700 dark:text-red-300 rounded-xl text-center shadow-lg border border-red-200 dark:border-red-800/50 animate-pulse">
            {{ error }}
          </div>
        </transition>
      </main>
      
      <TheFooter />
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import TheHeader from './components/TheHeader.vue'
import TheFooter from './components/TheFooter.vue'
import ResultCard from './components/ResultCard.vue'

export default {
  components: {
    TheHeader,
    TheFooter,
    ResultCard
  },
  
  setup() {
    const isDark = ref(false)
    
    const toggleTheme = () => {
      isDark.value = !isDark.value
      localStorage.setItem('theme', isDark.value ? 'dark' : 'light')
      updateTheme()
    }
    
    const updateTheme = () => {
      if (isDark.value) {
        document.documentElement.classList.add('dark')
      } else {
        document.documentElement.classList.remove('dark')
      }
    }
    
    onMounted(() => {
      const savedTheme = localStorage.getItem('theme')
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
      
      if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
        isDark.value = true
        updateTheme()
      }
    })
    
    const text = ref('')
    const selectedModel = ref('logistic')
    const isDropdownOpen = ref(false)
    const models = ref([
      { id: 'logistic', name: 'Logistic Regression', description: 'Fast and lightweight model' }
    ])
    const result = ref(null)
    const isLoading = ref(false)
    const error = ref(null)
    
    const selectedModelName = computed(() => {
      const model = models.value.find(m => m.id === selectedModel.value)
      return model ? model.name : 'Select a model'
    })
    
    const selectModel = (modelId) => {
      selectedModel.value = modelId
      isDropdownOpen.value = false
    }
    
    const predict = async () => {
      if (!text.value.trim()) return
      
      isLoading.value = true
      error.value = null
      result.value = null
      
      try {
        const response = await fetch('http://localhost:5000/api/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            text: text.value,
            model: selectedModel.value
          })
        })
        
        if (!response.ok) {
          throw new Error('Failed to get prediction')
        }
        
        result.value = await response.json()
      } catch (err) {
        error.value = 'Error: ' + (err.message || 'Failed to analyze text')
        console.error(err)
      } finally {
        isLoading.value = false
      }
    }
    
    const fetchModels = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/models')
        if (response.ok) {
          const data = await response.json()
          models.value = data.models
          if (models.value.length > 0) {
            selectedModel.value = models.value[0].id
          }
        }
      } catch (err) {
        console.error('Failed to fetch models:', err)
      }
    }
    
    const handleClickOutside = (event) => {
      if (isDropdownOpen.value && !event.target.closest('.relative')) {
        isDropdownOpen.value = false
      }
    }
    
    onMounted(() => {
      document.addEventListener('click', handleClickOutside)
      fetchModels()
    })
    
    onBeforeUnmount(() => {
      document.removeEventListener('click', handleClickOutside)
    })
    
    return {
      isDark,
      toggleTheme,
      
      text,
      selectedModel,
      models,
      isDropdownOpen,
      selectedModelName,
      selectModel,
      
      result,
      isLoading,
      error,
      predict
    }
  }
}
</script>

<style>
@keyframes float {
  0%, 100% {
    transform: translateY(0) rotate(0deg);
  }
  25% {
    transform: translateY(-15px) rotate(5deg);
  }
  50% {
    transform: translateY(5px) rotate(-5deg);
  }
  75% {
    transform: translateY(-10px) rotate(3deg);
  }
}

.particle {
  animation: float infinite linear;
}

.particle-1 { animation-duration: 25s; }
.particle-2 { animation-duration: 35s; }
.particle-3 { animation-duration: 40s; }
.particle-4 { animation-duration: 30s; }
.particle-5 { animation-duration: 45s; }
.particle-6 { animation-duration: 28s; }
.particle-7 { animation-duration: 38s; }
.particle-8 { animation-duration: 32s; }

.loader-ring {
  position: absolute;
  width: 100%;
  height: 100%;
  border: 2px solid rgba(255, 255, 255, 0.3);
  border-top-color: #ffffff;
  border-radius: 50%;
  animation: loader-spin 1s linear infinite;
}

@keyframes loader-spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

.fade-slide-up-enter-active,
.fade-slide-up-leave-active {
  transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}

.fade-slide-up-enter-from,
.fade-slide-up-leave-to {
  opacity: 0;
  transform: translateY(20px);
}

@keyframes dropdown {
  from {
    opacity: 0;
    transform: scaleY(0.8);
  }
  to {
    opacity: 1;
    transform: scaleY(1);
  }
}

.animate-dropdown {
  animation: dropdown 0.2s ease-out forwards;
}
</style>