<template>
  <div class="min-h-screen transition-colors duration-300" :class="{ 'dark': isDark, 'bg-gray-50 text-gray-900': !isDark, 'bg-gray-900 text-gray-100': isDark }">
    <div class="container mx-auto px-4 py-8 max-w-4xl">
      <TheHeader :isDark="isDark" @toggle-theme="toggleTheme" />
      
      <main class="my-8">
        <div class="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 transition-colors duration-300">
          <div class="mb-6">
            <label for="text-input" class="block mb-2 font-medium text-gray-700 dark:text-gray-300">
              Enter text to analyze:
            </label>
            <textarea 
              id="text-input" 
              v-model="text"
              placeholder="Paste or type text here..."
              rows="8"
              class="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-colors"
              :disabled="isLoading"
            ></textarea>
          </div>
          
          <div class="mb-6">
            <label class="block mb-2 font-medium text-gray-700 dark:text-gray-300">
              Select model:
            </label>
            <div class="relative">
              <button 
                type="button"
                @click="isDropdownOpen = !isDropdownOpen"
                class="w-full flex items-center justify-between px-4 py-3 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg text-left focus:outline-none focus:ring-2 focus:ring-primary-500 transition-colors"
              >
                <span>{{ selectedModelName }}</span>
                <svg 
                  xmlns="http://www.w3.org/2000/svg" 
                  class="h-5 w-5 text-gray-500 dark:text-gray-400 transition-transform duration-200" 
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
                class="absolute z-10 w-full mt-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg overflow-hidden"
              >
                <div class="max-h-60 overflow-y-auto">
                  <button
                    v-for="model in models"
                    :key="model.id"
                    @click="selectModel(model.id)"
                    class="w-full px-4 py-3 text-left hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
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
            class="w-full px-4 py-3 rounded-lg font-medium transition-all focus:outline-none focus:ring-2 focus:ring-offset-2 bg-primary-600 hover:bg-primary-700 text-white focus:ring-primary-500 dark:bg-primary-700 dark:hover:bg-primary-600 disabled:opacity-60 disabled:cursor-not-allowed"
          >
            <div class="flex items-center justify-center">
              <svg v-if="isLoading" class="animate-spin -ml-1 mr-2 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              {{ isLoading ? 'Analyzing...' : 'Analyze Text' }}
            </div>
          </button>
        </div>
        
        <ResultCard v-if="result" :result="result" class="mt-6" />
        
        <div v-if="error" class="mt-6 p-4 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 rounded-lg text-center">
          {{ error }}
        </div>
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