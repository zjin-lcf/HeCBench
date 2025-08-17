
/* This is a slight modification to the RA file format provided at
   http://github.com/davidssmith/ra
   that includes CUDA support for allocating the RA data area as pinned
   memory which in turn allows asynchronous kernel execution.
*/

#include <sycl/sycl.hpp>
#include "ra.h"

int
ra_read (ra_t *a, const char *path)
{
    int fd;
    uint64_t bytestoread, bytesleft, magic;
    fd = open(path, O_RDONLY);
    if (fd == -1)
        err(errno, "unable to open %s for reading", path);
    read(fd, &magic, sizeof(uint64_t));
    read(fd, &(a->flags), sizeof(uint64_t));
    read(fd, &(a->eltype), sizeof(uint64_t));
    read(fd, &(a->elbyte), sizeof(uint64_t));
    read(fd, &(a->size), sizeof(uint64_t));
    read(fd, &(a->ndims), sizeof(uint64_t));
    a->dims = (uint64_t*)malloc(a->ndims*sizeof(uint64_t));
    read(fd, a->dims, a->ndims*sizeof(uint64_t));
    a->data = (uint8_t*)malloc(a->size);
    if (a->data == NULL) err(errno, "unable to allocate memory for data");
    uint8_t *data_cursor = a->data;

    bytesleft = a->size;
    while (bytesleft > 0) {
        bytestoread = bytesleft < RA_MAX_BYTES ? bytesleft : RA_MAX_BYTES;
        read(fd, data_cursor, bytestoread);
        data_cursor += bytestoread;
        bytesleft -= bytestoread;
    }
    close(fd);
    return 0;
}

int
ra_write (ra_t *a, const char *path)
{
    int fd;
    uint64_t bytesleft, bufsize;
    uint8_t *data_in_cursor;
    fd = open(path, O_WRONLY|O_TRUNC|O_CREAT,0644);
    if (fd == -1)
        err(errno, "unable to open %s for writing", path);
    /* write the easy stuff */
    write(fd, &RA_MAGIC_NUMBER, sizeof(uint64_t));
    write(fd, &(a->flags), sizeof(uint64_t));
    write(fd, &(a->eltype), sizeof(uint64_t));
    write(fd, &(a->elbyte), sizeof(uint64_t));
    write(fd, &(a->size), sizeof(uint64_t));
    write(fd, &(a->ndims), sizeof(uint64_t));
    write(fd, a->dims, a->ndims*sizeof(uint64_t));

    bytesleft = a->size;
    data_in_cursor = a->data;

    bufsize = bytesleft < RA_MAX_BYTES ? bytesleft : RA_MAX_BYTES;
    while (bytesleft > 0) {
        write(fd, data_in_cursor, bufsize);
        data_in_cursor += bufsize / sizeof(uint8_t);
        bytesleft -= bufsize;
    }

    close(fd);
    return 0;
}

void
ra_free (ra_t *a)
{
    free(a->data);
    free(a->dims);
}
