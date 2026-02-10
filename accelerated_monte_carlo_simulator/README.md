In a european options market, you can't exercise your options trade early, As opposed to american markets
This makes monte carlo simulation using GPUs much simpler for EU options

I am letting the threads use a stride due to the potential high number of paths in this calculation, we can store less RNG states, and ensures we have just enough threads to staurate gpu