// Example: How your website could consume the improved babyState API

class BabyLLMState {
  constructor() {
    this.lastData = null;
    this.updateInterval = null;
    this.isPolling = false;
  }

  async fetchState() {
    try {
      const response = await fetch('http://localhost:4420/api/babystate', {
        method: 'GET',
        mode: 'cors'
      });
      
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const data = await response.json();
      this.handleNewData(data);
      return data;
    } catch (error) {
      console.warn('Failed to fetch babyLLM state:', error);
      return null;
    }
  }

  handleNewData(data) {
    const meta = data.meta || {};
    
    // Check if data is stale
    if (meta.isStale) {
      console.warn('BabyLLM data is stale!', `Age: ${meta.dataAge}s`);
      // Maybe show a "disconnected" indicator
    }

    // Extract the actual state (without meta)
    const { meta: _, ...babyState } = data;
    
    // Use the state for animations/display
    this.updateVisuals(babyState);
    
    this.lastData = data;
  }

  updateVisuals(state) {
    // Example: Update baby's color
    document.documentElement.style.setProperty('--baby-r', state.R);
    document.documentElement.style.setProperty('--baby-g', state.G);
    document.documentElement.style.setProperty('--baby-b', state.B);
    
    // Example: Scale animations based on metabolicRate
    const animSpeed = Math.max(0.1, state.metabolicRate || 0.1);
    document.documentElement.style.setProperty('--anim-speed', `${animSpeed}s`);
    
    // Example: Visual indicators for neural activity
    document.querySelector('.cerebral-load')?.style.setProperty('opacity', state.cerebralLoad);
    document.querySelector('.dream-intensity')?.style.setProperty('transform', 
      `scale(${1 + (state.dreamIntensity || 0) * 0.1})`);
  }

  startPolling(intervalMs = 1000) {
    if (this.isPolling) return;
    
    this.isPolling = true;
    this.updateInterval = setInterval(() => {
      this.fetchState();
    }, intervalMs);
    
    // Initial fetch
    this.fetchState();
  }

  stopPolling() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
    this.isPolling = false;
  }
}

// Usage:
const babyState = new BabyLLMState();
babyState.startPolling(500); // Poll every 500ms
