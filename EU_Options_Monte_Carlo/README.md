In a european options market, you can't exercise your options trade early, As opposed to american markets
This makes monte carlo simulation using GPUs much simpler for EU options

I am letting the threads use a stride due to the potential high number of paths in this calculation, we can store less RNG states, and ensures we have just enough threads to staurate gpu

Benchmark (CPU loop vs GPU kernel, 1,000,000 paths):
```
NUM_PATHS      : 1000000
threads/block  : 256
blocks         : 3907
num_states     : 1000192
CPU price      : 2.48008
GPU price      : 2.4692
CPU time (ms)  : 75.724
CPU loop (ms)  : 75.7211
GPU kernel (ms): 0.243712
Speedup        : 310.699x
```

Speedup:
