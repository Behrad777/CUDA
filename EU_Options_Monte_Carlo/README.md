In a european options market, you can't exercise your options trade early, As opposed to american markets
This makes monte carlo simulation using GPUs much simpler for EU options

I am letting the threads use a stride due to the potential high number of paths in this calculation, we can store less RNG states, and ensures we have just enough threads to staurate gpu

Benchmark (GPU kernel, 1,000,000 paths):
```

NOTICE: Existing SQLite export found: result.sqlite
        It is assumed file was previously exported from: result.nsys-rep
        Consider using --force-export=true if needed.

Processing [result.sqlite] with [/home/bassa/cuda-12.4/nsight-systems-2023.4.4/host-linux-x64/reports/nvtx_sum.py]... 
SKIPPED: result.sqlite does not contain NV Tools Extension (NVTX) data.

Processing [result.sqlite] with [/home/bassa/cuda-12.4/nsight-systems-2023.4.4/host-linux-x64/reports/osrt_sum.py]... 

 ** OS Runtime Summary (osrt_sum):

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)     Med (ns)    Min (ns)   Max (ns)    StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  -----------  --------  -----------  ------------  ----------------------
     79.9      333,740,784         12  27,811,732.0  4,684,577.5     1,343  233,289,048  65,884,439.2  poll                  
     19.2       80,331,846        479     167,707.4     11,962.0       822   12,271,504     693,025.7  ioctl                 
      0.6        2,371,219         25      94,848.8      8,185.0     6,072    1,694,684     336,204.8  mmap64                
      0.1          358,596          9      39,844.0     40,245.0    27,863       48,392       6,275.3  sem_timedwait         
      0.1          229,581         43       5,339.1      5,019.0     2,866        8,817       1,122.7  open64                
      0.0          168,916         14      12,065.4      5,756.0     2,835       81,724      20,426.0  mmap                  
      0.0          156,979         27       5,814.0      4,529.0     1,192       17,403       3,950.9  fopen                 
      0.0          108,455          3      36,151.7     35,187.0    27,281       45,987       9,390.2  pthread_create        
      0.0           62,749         31       2,024.2         40.0        30       61,656      11,067.2  fgets                 
      0.0           42,830         21       2,039.5      1,874.0     1,312        3,827         650.8  fclose                
      0.0           37,629         14       2,687.8      2,715.0     1,823        3,927         446.6  read                  
      0.0           29,657          6       4,942.8      5,135.0     2,394        7,284       1,668.2  open                  
      0.0           24,626          7       3,518.0      3,556.0     2,805        3,917         414.3  munmap                
      0.0           22,813         11       2,073.9      2,164.0       862        2,645         474.5  write                 
      0.0           22,470         57         394.2        401.0       270          781          81.2  fcntl                 
      0.0           18,806          3       6,268.7      5,681.0     2,755       10,370       3,841.4  pipe2                 
      0.0           17,923          2       8,961.5      8,961.5     5,760       12,163       4,527.6  fread                 
      0.0           10,199          2       5,099.5      5,099.5     3,677        6,522       2,011.7  socket                
      0.0            6,813          1       6,813.0      6,813.0     6,813        6,813           0.0  connect               
      0.0            5,851          3       1,950.3      1,953.0     1,924        1,974          25.1  pthread_cond_broadcast
      0.0            4,692         64          73.3         90.0        20          431          59.4  pthread_mutex_trylock 
      0.0            2,525          2       1,262.5      1,262.5     1,233        1,292          41.7  fwrite                
      0.0            2,446          7         349.4        351.0       311          400          36.9  dup                   
      0.0            1,523          1       1,523.0      1,523.0     1,523        1,523           0.0  bind                  
      0.0              571          1         571.0        571.0       571          571           0.0  listen                

Processing [result.sqlite] with [/home/bassa/cuda-12.4/nsight-systems-2023.4.4/host-linux-x64/reports/cuda_api_sum.py]... 

 ** CUDA API Summary (cuda_api_sum):

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)            Name         
 --------  ---------------  ---------  ------------  -----------  --------  ----------  ------------  ----------------------
     88.0       89,837,254          5  17,967,450.8      3,958.0     3,396  89,821,966  40,167,895.1  cudaMemcpyToSymbol    
     10.4       10,594,855          2   5,297,427.5  5,297,427.5   240,824  10,354,031   7,151,117.2  cudaDeviceSynchronize 
      1.1        1,141,351          2     570,675.5    570,675.5    77,987   1,063,364     696,766.8  cudaFree              
      0.3          351,843          1     351,843.0    351,843.0   351,843     351,843           0.0  cudaMemcpy            
      0.1          139,053          2      69,526.5     69,526.5    67,618      71,435       2,699.0  cudaMalloc            
      0.1           66,005          3      22,001.7     29,065.0     3,166      33,774      16,481.2  cudaLaunchKernel      
      0.0            1,693          1       1,693.0      1,693.0     1,693       1,693           0.0  cuModuleGetLoadingMode

Processing [result.sqlite] with [/home/bassa/cuda-12.4/nsight-systems-2023.4.4/host-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)                             Name                           
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----------------------------------------------------------
     95.6       10,353,487          1  10,353,487.0  10,353,487.0  10,353,487  10,353,487          0.0  init_rng_kernel(curandStateXORWOW *, unsigned long)       
      4.4          479,199          2     239,599.5     239,599.5     239,392     239,807        293.4  monte_carlo_kernel(float *, curandStateXORWOW *, int, int)

Processing [result.sqlite] with [/home/bassa/cuda-12.4/nsight-systems-2023.4.4/host-linux-x64/reports/cuda_gpu_mem_time_sum.py]... 

 ** CUDA GPU MemOps Summary (by Time) (cuda_gpu_mem_time_sum):

 Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)           Operation          
 --------  ---------------  -----  ---------  ---------  --------  --------  -----------  ----------------------------
     99.4          271,615      1  271,615.0  271,615.0   271,615   271,615          0.0  [CUDA memcpy Device-to-Host]
      0.6            1,728      5      345.6      352.0       320       352         14.3  [CUDA memcpy Host-to-Device]

Processing [result.sqlite] with [/home/bassa/cuda-12.4/nsight-systems-2023.4.4/host-linux-x64/reports/cuda_gpu_mem_size_sum.py]... 

 ** CUDA GPU MemOps Summary (by Size) (cuda_gpu_mem_size_sum):

 Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation          
 ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
      4.000      1     4.000     4.000     4.000     4.000        0.000  [CUDA memcpy Device-to-Host]
      0.000      5     0.000     0.000     0.000     0.000        0.000  [CUDA memcpy Host-to-Device]

```