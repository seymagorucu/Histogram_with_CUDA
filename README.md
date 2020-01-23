# Histogram_with_CUDA

Firstly, I executed  basic histogram and kogge stone scan algorithm. Basic histogram function doesn’t use shared memory. Also  atomicAdd provide read-modify-write operation performed by single hardware instruction on a memory location address.
We assign each thread to enhance the content of an scan element. I have defined the size of a section as compile-time constant SECTION_SIZE. I used SECTION_SIZE as the block size of the kernel initialization, so I've had an equal number of threads and partition elements. Then made final adjustments to these cross-sectional scanning results for large input sequences. Histo[] array  has all the threads in the block to load array elements together into a common memory array scan[] . I used a barrier synchronization to allow all threads to repeat their current insertions in before starting the next iteration.
 Also  I find cdf min  to calculate histogram equalization. At the end of the kernel, each thread writes its result to the assigned output array scanning[]. Then histogram equalize is calculated.
 My device is GeForce 940MX warp size is 32. My  SECTION_SIZE is 256.  256 / 32 = 8 warps are used.
 
 
 Then, I executed private histogram and brent kunt scan functions. These two algorithms run faster than the previous one. The private histogram used shared memory. Private histogram provides much less contention and serialization for accessing both private copies and the final copy. Therefore, it improves performance.Since the Brent-Kung algorithm always uses consecutive threads in each iteration, the control deviation problem does not occur until the number of active threads falls below the warp size. This can increase the efficiency of the algorithm. Then I find cdf min  to calculate histogram equalization. At the end of the kernel, each thread writes its result to the assigned output array scanning[]. Then histogram equalize is calculated.
 
 
So in the second part, using more efficient codes for both histogram and scan has accelerated the process. 
