#ifndef DSM_COMMON_H
#define DSM_COMMON_H

typedef unsigned int uint;

extern "C" void dsm_sm2sm_latency(void *d_Data, uint arraySize, uint cluster_size);

extern "C" void dsm_sm2sm_thrpt(void *d_Data, uint arraySize, uint cluster_size, uint block_num, uint block_size, uint repeat_times);

#endif
