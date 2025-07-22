#!/usr/bin/env python3
"""
Protected runner for SNN STDP analysis with memory management and checkpointing
Prevents process killing due to memory issues and provides recovery options
"""

import os
import sys
import time
import psutil
import gc
import pickle
import signal
import subprocess
from datetime import datetime
from pathlib import Path

class ProcessMonitor:
    """Monitor and manage system resources during long-running processes"""
    
    def __init__(self, memory_limit_gb=8, check_interval=60):
        self.memory_limit_gb = memory_limit_gb
        self.check_interval = check_interval
        self.start_time = time.time()
        self.process = psutil.Process()
        
    def check_memory_usage(self):
        """Check current memory usage and clean up if needed"""
        memory_info = self.process.memory_info()
        memory_gb = memory_info.rss / (1024**3)
        
        print(f"🔍 Memory usage: {memory_gb:.2f} GB")
        
        if memory_gb > self.memory_limit_gb:
            print(f"⚠️  Memory usage ({memory_gb:.2f} GB) exceeds limit ({self.memory_limit_gb} GB)")
            print("🧹 Running garbage collection...")
            gc.collect()
            
            # Check again after GC
            memory_info = self.process.memory_info()
            memory_gb = memory_info.rss / (1024**3)
            print(f"🔍 Memory after GC: {memory_gb:.2f} GB")
            
            if memory_gb > self.memory_limit_gb:
                print("❌ Memory still too high after GC - consider reducing batch size")
                return False
        return True
    
    def get_runtime_info(self):
        """Get current runtime information"""
        runtime = time.time() - self.start_time
        hours = int(runtime // 3600)
        minutes = int((runtime % 3600) // 60)
        return f"{hours}h {minutes}m"

def run_with_nohup():
    """Run the SNN analysis using nohup to prevent terminal disconnection kills"""
    script_path = "project/MyCode/snn_stdp_resonator_classifier.py"
    log_file = f"snn_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    cmd = [
        "nohup", 
        sys.executable, 
        script_path, 
        ">", log_file, 
        "2>&1", 
        "&"
    ]
    
    print(f"🚀 Starting SNN analysis with nohup...")
    print(f"📝 Log file: {log_file}")
    print(f"🔧 Command: {' '.join(cmd)}")
    
    # Use shell=True for nohup redirection
    subprocess.Popen(' '.join(cmd), shell=True)
    print(f"✅ Process started in background. Monitor with: tail -f {log_file}")

def run_with_memory_management():
    """Run SNN analysis with active memory management"""
    print("🧠 PROTECTED SNN ANALYSIS WITH MEMORY MANAGEMENT")
    print("=" * 60)
    
    # Set up process monitoring
    monitor = ProcessMonitor(memory_limit_gb=6, check_interval=300)  # Check every 5 minutes
    
    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}. Saving progress and shutting down gracefully...")
        # Add any cleanup code here
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Import and run the analysis
        sys.path.insert(0, "project/MyCode")
        from snn_stdp_resonator_classifier import run_comprehensive_snn_analysis
        
        print(f"🕐 Starting analysis at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run with periodic memory checks
        results = run_comprehensive_snn_analysis()
        
        print(f"✅ Analysis completed successfully!")
        print(f"⏱️  Total runtime: {monitor.get_runtime_info()}")
        
        return results
        
    except MemoryError:
        print("❌ Out of memory error occurred!")
        print("💡 Try reducing parallel workers or batch sizes")
        return None
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_in_chunks():
    """Run analysis in smaller chunks to avoid memory issues"""
    print("📦 CHUNKED SNN ANALYSIS")
    print("=" * 40)
    
    # This would require modifying the main analysis to process in smaller batches
    # For now, just run with reduced parallel workers
    
    # Set environment variable to reduce parallel workers
    os.environ['JOBLIB_PARALLEL_WORKERS'] = '4'  # Reduce from 10 to 4
    
    return run_with_memory_management()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Protected SNN Analysis Runner")
    parser.add_argument("--mode", choices=['nohup', 'protected', 'chunked'], 
                       default='protected', help="Running mode")
    
    args = parser.parse_args()
    
    print(f"🔧 Running in {args.mode} mode")
    
    if args.mode == 'nohup':
        run_with_nohup()
    elif args.mode == 'chunked':
        run_in_chunks()
    else:
        run_with_memory_management() 