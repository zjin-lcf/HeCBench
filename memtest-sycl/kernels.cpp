#define TYPE unsigned long
#define MAX_ERR_RECORD_COUNT 10
#define MOD_SZ 20
#define BLOCKSIZE (1024*1024)

#define RECORD_ERR(err, p, expect, current) do{ \
    auto atomic_obj_ref = ext::oneapi::atomic_ref<unsigned int, \
      ext::oneapi::memory_order::relaxed, \
      ext::oneapi::memory_scope::device, \
      access::address_space::global_space> (err[0]); \
    unsigned int idx = atomic_obj_ref.fetch_add(1u); \
    idx = idx % MAX_ERR_RECORD_COUNT; \
    err_addr[idx] = (unsigned long)p; \
    err_expect[idx] = (unsigned long)expect; \
    err_current[idx] = (unsigned long)current; \
    err_second_read[idx] = (unsigned long)(*p); \
  } while(0)

//each thread is responsible for 1 BLOCKSIZE each time
void kernel0_write(nd_item<1> &item, char* ptr, unsigned long size)
{
  unsigned long* buf = (unsigned long*)ptr;
  int idx = item.get_global_id(0);
  unsigned long n = size/BLOCKSIZE;
  int total_num_threads = item.get_global_range(0);
  
  for(int i = idx; i < n; i += total_num_threads) {
    unsigned long * start_p= (unsigned long*)(ptr + i*BLOCKSIZE);
    unsigned long * end_p = (unsigned long*)(ptr + (i+1)*BLOCKSIZE);
    unsigned long * p =start_p;
    unsigned int pattern = 1;
    unsigned int mask = 8;
      
    *p = pattern;
    pattern = (pattern << 1);
    while(p < end_p){
      p = (unsigned long*) (((unsigned long)start_p)|mask);
      
      if(p == start_p) {
	mask = (mask << 1);
	if (mask == 0){
	  break;
	}
	continue;
      }
      
      if (p >= end_p) break;
      
      *p = pattern;
      pattern = pattern <<1;
      mask = (mask << 1);
      if (mask == 0) break;
    }
  }
}

void kernel0_read(nd_item<1> &item,
                  const char* ptr, unsigned long size,
                  unsigned int* err_count,
                  unsigned long* err_addr,
                  unsigned long* err_expect,
                  unsigned long* err_current,
                  unsigned long* err_second_read)
{
  int idx = item.get_global_id(0);
  unsigned long n = size/BLOCKSIZE;
  int total_num_threads = item.get_global_range(0);

  for(int i = idx; i < n; i += total_num_threads) {
    unsigned long * start_p= (unsigned long*)(ptr + i*BLOCKSIZE);
    unsigned long * end_p = (unsigned long*)(ptr + (i+1)*BLOCKSIZE);
    unsigned long * p =start_p;
    unsigned int pattern = 1;
    unsigned int mask = 8;
      
    if (*p != pattern){
      RECORD_ERR(err_count, p, pattern, *p);
    }
    
    pattern = (pattern << 1);
    while(p< end_p){
      p = ( unsigned long*)( ((unsigned long)start_p)|mask);
      
      if(p == start_p){
	mask = (mask << 1);
	if (mask == 0){
	  break;
	}
	continue;
      }
      
      if (p >= end_p){
	break;
      }
      
      if (*p != pattern){
	RECORD_ERR(err_count, p, pattern, *p);
      }

      pattern = pattern <<1;
      mask = (mask << 1);
      if (mask == 0){
	break;
      }
    }
  }
}


void kernel1_write(nd_item<1> &item, char* ptr, unsigned long size)
{
  int idx = item.get_global_id(0);

  unsigned long* buf = ( unsigned long*)ptr;
  unsigned long n = size/sizeof(unsigned long);
  int total_num_threads = item.get_global_range(0);
  
  for(int i = idx; i < n; i += total_num_threads)
    buf[i] = (unsigned long)(buf+i);
}

void kernel1_read(nd_item<1> &item,
                  const char* ptr, unsigned long size,
                  unsigned int* err_count,
                  unsigned long* err_addr,
                  unsigned long* err_expect,
                  unsigned long* err_current,
                  unsigned long* err_second_read)
{
  unsigned long* buf = ( unsigned long*)ptr;
  int idx = item.get_global_id(0);
  unsigned long n = size/sizeof(unsigned long);
  int total_num_threads = item.get_global_range(0);
  
  for(int i = idx; i < n; i += total_num_threads) {
    if(buf[i] != (unsigned long)(buf+i))
      RECORD_ERR(err_count, &buf[i], (buf+i), buf[i]);
  }
}

// kernels called by multiple tests
void kernel_write(nd_item<1> &item, char* ptr, unsigned long size, TYPE p1)
{
   TYPE* buf = (TYPE*)ptr;
  int idx = item.get_global_id(0);
  unsigned long n = size/sizeof(TYPE);
  int total_num_threads = item.get_global_range(0);
  
  for(int i = idx; i < n; i+= total_num_threads){
    buf[i] = p1;
  } 
}


void kernel_read_write(
nd_item<1> &item,
char* ptr, unsigned long size, TYPE p1, TYPE p2,
unsigned int* err_count,
unsigned long* err_addr,
unsigned long* err_expect,
unsigned long* err_current,
unsigned long* err_second_read)
{
  TYPE* buf = (TYPE*) ptr;
  int idx = item.get_global_id(0);
  unsigned long n =  size/sizeof(TYPE);
  int total_num_threads = item.get_global_range(0);
  TYPE localp;
  
  for(int i = idx;i < n; i += total_num_threads){
    
    localp = buf[i];
    
    if (localp != p1){
      RECORD_ERR(err_count, &buf[i], p1, localp);
    }
    
    buf[i] = p2;
  }
}

void kernel_read(nd_item<1> &item,
                 char* ptr, unsigned long size, TYPE p1,
                 unsigned int* err_count,
                 unsigned long* err_addr,
                 unsigned long* err_expect,
                 unsigned long* err_current,
                 unsigned long* err_second_read)
{
  TYPE* buf = (TYPE*) ptr;
  int idx = item.get_global_id(0);
  unsigned long n =  size/sizeof(TYPE);
  int total_num_threads = item.get_global_range(0);
  TYPE localp;
  
  for(int i = idx;i < n; i += total_num_threads){
    localp = buf[i];
    if (localp != p1){
      RECORD_ERR(err_count, &buf[i], p1, localp);
    }
  }
}
