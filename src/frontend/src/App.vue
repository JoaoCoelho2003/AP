<template>
  <div class="min-h-screen transition-all duration-500 relative overflow-hidden" 
       :class="{ 'dark': isDark, 'bg-gray-50 text-gray-900': !isDark, 'bg-gray-900 text-gray-100': isDark }">
    <div class="fixed inset-0 pointer-events-none">
      <canvas ref="neuralCanvas" class="absolute inset-0 w-full h-full"></canvas>
      
      <div class="absolute inset-0 bg-gradient-to-br from-primary-100/30 to-transparent dark:from-primary-900/20 dark:to-transparent"></div>
      <div class="absolute -top-[40%] -right-[30%] w-[80%] h-[80%] rounded-full bg-gradient-to-br from-primary-200/20 to-purple-300/20 dark:from-primary-800/10 dark:to-purple-900/10 blur-3xl"></div>
      <div class="absolute -bottom-[40%] -left-[30%] w-[80%] h-[80%] rounded-full bg-gradient-to-tr from-blue-200/20 to-primary-300/20 dark:from-blue-900/10 dark:to-primary-800/10 blur-3xl"></div>
    </div>

    <div class="w-full relative">
      <TheHeader :isDark="isDark" @toggle-theme="toggleTheme" class="w-full" />
      
      <main class="container mx-auto px-4 py-8">
        <div class="flex flex-col lg:flex-row gap-8">
          <div class="w-full lg:w-1/2">
            <div class="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-2xl shadow-xl p-8 transition-all duration-300 transform hover:shadow-2xl min-h-[500px]">
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
          </div>
          
          <div class="w-full lg:w-1/2">
            <div class="min-h-[500px]">
              <transition name="fade" mode="out-in">
                <ResultCard v-if="result" :result="result" key="result" />
                <div v-else key="empty" class="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-2xl shadow-xl p-8 transition-all duration-300 transform hover:shadow-2xl h-full flex flex-col items-center justify-center text-center">
                  <div class="w-24 h-24 mb-6 rounded-full bg-primary-100 dark:bg-primary-900/30 flex items-center justify-center">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 text-primary-500 dark:text-primary-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                    </svg>
                  </div>
                  <h3 class="text-xl font-bold text-gray-800 dark:text-gray-200 mb-2">No Analysis Yet</h3>
                  <p class="text-gray-600 dark:text-gray-400 max-w-md mb-6">
                    Enter your text in the input field and click "Analyze Text" to see AI detection results here.
                  </p>
                  <div class="flex flex-wrap justify-center gap-3">
                    <div class="px-4 py-2 rounded-lg bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 text-sm">
                      Fast Analysis
                    </div>
                    <div class="px-4 py-2 rounded-lg bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 text-sm">
                      High Accuracy
                    </div>
                    <div class="px-4 py-2 rounded-lg bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300 text-sm">
                      Detailed Results
                    </div>
                  </div>
                </div>
              </transition>
            </div>
            
            <transition name="fade">
              <div v-if="error" class="mt-8 p-5 bg-red-100/80 dark:bg-red-900/30 backdrop-blur-sm text-red-700 dark:text-red-300 rounded-xl text-center shadow-lg border border-red-200 dark:border-red-800/50 animate-pulse">
                {{ error }}
              </div>
            </transition>
          </div>
        </div>
      </main>
      
      <TheFooter class="w-full" />
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
    const neuralCanvas = ref(null)
    let animationFrame = null
    
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
        
        const data = await response.json()
        setTimeout(() => {
          result.value = data
          isLoading.value = false
        }, 0)
      } catch (err) {
        error.value = 'Error: ' + (err.message || 'Failed to analyze text')
        console.error(err)
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
    
    const initNeuralNetwork = () => {
      const canvas = neuralCanvas.value
      if (!canvas) return

      const ctx = canvas.getContext('2d')
      if (!ctx) return

      const nodes = []
      const connections = []
      const nodeCount = Math.min(Math.floor(window.innerWidth / 100), 20)
      
      const startX = Math.random() * canvas.width * 0.8
      const startY = Math.random() * canvas.height * 0.8
      
      for (let i = 0; i < nodeCount; i++) {
        const randomOffsetX = (Math.random() - 0.5) * canvas.width * 0.4
        const randomOffsetY = (Math.random() - 0.5) * canvas.height * 0.4
        
        nodes.push({
          x: startX + randomOffsetX,
          y: startY + randomOffsetY,
          radius: Math.random() * 2 + 2,
          vx: (Math.random() - 0.5) * 1.2,
          vy: (Math.random() - 0.5) * 1.2,
          color: isDark.value ? 
            `rgba(${100 + Math.random() * 155}, ${100 + Math.random() * 155}, ${200 + Math.random() * 55}, 0.5)` : 
            `rgba(${50 + Math.random() * 155}, ${100 + Math.random() * 155}, ${200 + Math.random() * 55}, 0.5)`
        })
      }
      
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          if (Math.random() > 0.85) {
            connections.push({
              from: i,
              to: j,
              width: Math.random() * 0.5 + 0.2,
              pulseSpeed: Math.random() * 0.05 + 0.01,
              pulseOffset: Math.random() * Math.PI * 2,
              color: isDark.value ? 'rgba(100, 150, 255, 0.15)' : 'rgba(70, 130, 230, 0.1)'
            })
          }
        }
      }
      
      const drawNodes = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height)
      
        connections.forEach(connection => {
          const fromNode = nodes[connection.from]
          const toNode = nodes[connection.to]
      
          const dx = fromNode.x - toNode.x
          const dy = fromNode.y - toNode.y
          const distance = Math.sqrt(dx * dx + dy * dy)
      
          if (distance < canvas.width / 3) {
            const time = Date.now() * connection.pulseSpeed + connection.pulseOffset
            const pulse = Math.sin(time) * 0.5 + 0.5
      
            ctx.beginPath()
            ctx.moveTo(fromNode.x, fromNode.y)
            ctx.lineTo(toNode.x, toNode.y)
            ctx.strokeStyle = connection.color
            ctx.lineWidth = connection.width * pulse
            ctx.stroke()
      
            const pulsePosition = (Math.sin(time / 2) + 1) / 2
            const pulseX = fromNode.x + (toNode.x - fromNode.x) * pulsePosition
            const pulseY = fromNode.y + (toNode.y - fromNode.y) * pulsePosition
      
            ctx.beginPath()
            ctx.arc(pulseX, pulseY, connection.width * 2, 0, Math.PI * 2)
            ctx.fillStyle = isDark.value ? 'rgba(120, 180, 255, 0.8)' : 'rgba(70, 130, 230, 0.6)'
            ctx.fill()
          }
        })
      
        nodes.forEach(node => {
          node.x += node.vx
          node.y += node.vy
      
          if (node.x < 0 || node.x > canvas.width) node.vx *= -1
          if (node.y < 0 || node.y > canvas.height) node.vy *= -1
      
          ctx.beginPath()
          ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2)
          ctx.fillStyle = node.color
          ctx.fill()
        })
      
        animationFrame = requestAnimationFrame(drawNodes)
      }
      
      const resizeCanvas = () => {
        canvas.width = window.innerWidth
        canvas.height = window.innerHeight
        drawNodes()
      }
      
      resizeCanvas()
      
      window.addEventListener('resize', resizeCanvas)

      return () => {
        if (animationFrame) {
          cancelAnimationFrame(animationFrame)
        }
        window.removeEventListener('resize', resizeCanvas)
      }
    }
    
    onMounted(() => {
      const savedTheme = localStorage.getItem('theme')
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
      
      if (savedTheme === 'dark' || (!savedTheme && prefersDark)) {
        isDark.value = true
        updateTheme()
      }
      
      document.addEventListener('click', handleClickOutside)
      fetchModels()
      
      setTimeout(() => {
        if (neuralCanvas.value) {
          initNeuralNetwork()
        }
      }, 100)
    })
    
    onBeforeUnmount(() => {
      document.removeEventListener('click', handleClickOutside)
      if (animationFrame) {
        cancelAnimationFrame(animationFrame)
      }
      window.removeEventListener('resize', () => {})
    })
    
    return {
      isDark,
      toggleTheme,
      neuralCanvas,
      
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

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
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

html {
  scroll-behavior: smooth;
}

.hide-scrollbar::-webkit-scrollbar {
  display: none;
}

.hide-scrollbar {
  -ms-overflow-style: none;
  scrollbar-width: none;
}

@media (max-width: 1023px) {
  .container {
    padding-left: 1rem;
    padding-right: 1rem;
  }
}

@media (min-width: 1024px) {
  .container {
    padding-left: 2rem;
    padding-right: 2rem;
  }
}
</style>