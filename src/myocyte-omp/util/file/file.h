#define fp float

//  WRITE FUNCTION

void write_file(
    const char* filename,
    fp* input, 
    int data_rows, 
    int data_cols, 
    int major,
    int data_range);

//  READ FUNCTION

void read_file(
    const char* filename,
    fp* input,
    int data_rows, 
    int data_cols,
    int major);

