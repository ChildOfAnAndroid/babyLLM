# v1.3
"""
Performance monitoring and health checks for babyLLM
Tracks system health and performance metrics
"""
import time
import psutil
import asyncio
from collections import defaultdict, deque
from typing import Dict, List, Optional
from .logger import logger

class PerformanceMonitor:
    """System performance monitoring and health checks"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics = defaultdict(lambda: deque(maxlen=history_size))
        self.alerts = defaultdict(list)
        self.health_checks = {}
        self.start_time = time.time()
    
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[float] = None):
        """Record a performance metric"""
        if timestamp is None:
            timestamp = time.time()
        
        self.metrics[metric_name].append((timestamp, value))
    
    def get_metric_average(self, metric_name: str, last_n: int = 100) -> Optional[float]:
        """Get average of last N metric values"""
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        recent_values = list(self.metrics[metric_name])[-last_n:]
        return sum(val for _, val in recent_values) / len(recent_values)
    
    def add_health_check(self, name: str, check_func, critical: bool = False):
        """Add a health check function"""
        self.health_checks[name] = {
            'func': check_func,
            'critical': critical,
            'last_run': None,
            'last_result': None
        }
    
    async def run_health_checks(self) -> Dict[str, bool]:
        """Run all health checks"""
        results = {}
        critical_failures = []
        
        for name, check in self.health_checks.items():
            try:
                start_time = time.time()
                result = await check['func']() if asyncio.iscoroutinefunction(check['func']) else check['func']()
                check_time = time.time() - start_time
                
                check['last_run'] = time.time()
                check['last_result'] = result
                results[name] = result
                
                self.record_metric(f"health_check_{name}_time", check_time)
                
                if not result:
                    logger.warn("HEALTH_CHECK", f"Health check '{name}' failed")
                    if check['critical']:
                        critical_failures.append(name)
                
            except Exception as e:
                logger.error("HEALTH_CHECK", f"Health check '{name}' crashed: {e}")
                results[name] = False
                if check['critical']:
                    critical_failures.append(name)
        
        if critical_failures:
            logger.emergency("HEALTH_CHECK", f"Critical health checks failed: {critical_failures}")
        
        return results
    
    def get_system_stats(self) -> Dict[str, float]:
        """Get current system statistics"""
        try:
            process = psutil.Process()
            stats = {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'uptime_hours': (time.time() - self.start_time) / 3600,
                'thread_count': process.num_threads(),
            }
            
            # Record metrics
            for key, value in stats.items():
                self.record_metric(key, value)
            
            return stats
        except Exception as e:
            logger.error("SYSTEM_STATS", f"Failed to get system stats: {e}")
            return {}
    
    def check_performance_degradation(self) -> List[str]:
        """Check for performance degradation patterns"""
        warnings = []
        
        # Check memory growth
        memory_avg_recent = self.get_metric_average('memory_mb', 50)
        memory_avg_old = self.get_metric_average('memory_mb', 500)
        
        if memory_avg_recent and memory_avg_old and memory_avg_recent > memory_avg_old * 1.5:
            warnings.append("Memory usage increasing significantly")
        
        # Check CPU spikes
        cpu_avg = self.get_metric_average('cpu_percent', 20)
        if cpu_avg and cpu_avg > 80:
            warnings.append("High CPU usage detected")
        
        return warnings

# Global performance monitor
perf_monitor = PerformanceMonitor()
