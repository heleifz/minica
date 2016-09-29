
cpdef int max_pooling_batch(double [:,:,:,:] src, double [:,:,:,:] dst, int [:,:,:,:] max_ind,
                            int win_h, int win_w, int stride_h, int stride_w):
    cdef:
        int num = src.shape[0]
        int ch = src.shape[1]
        int h_src = src.shape[2]
        int w_src = src.shape[3]
        int h_dst = dst.shape[2]
        int w_dst = dst.shape[3]
        int i, j, k, l, m, n
        int dst_ind = 0, src_ind = 0
        double *src_ptr = &src[0][0][0][0]
        double *dst_ptr = &dst[0][0][0][0]
        int *ind_ptr = &max_ind[0][0][0][0]
        double max_val
        int current_max_ind
        double tmp
        int max_h = h_src - win_h + 1
        int max_w = w_src - win_w + 1


    for i in range(num):
        for j in range(ch):
            for k in range(0, max_h, stride_h):
                for l in range(0, max_w, stride_w):
                    src_ind = i * ch * h_src * w_src + j * h_src * w_src + k * w_src + l
                    max_val = -1e20
                    for m in range(win_h):
                        for n in range(win_w):
                            tmp = src_ptr[src_ind + n]
                            if tmp > max_val:
                                max_val = tmp
                                current_max_ind = src_ind + n
                        src_ind += w_src
                    dst_ptr[dst_ind] = max_val
                    ind_ptr[dst_ind] = current_max_ind
                    dst_ind += 1
    return 0

cpdef int backprop_for_max_pooling(double [:,:,:,:] src, double [:,:,:,:] dst, int [:,:,:,:] max_ind):
    cdef:
        int num = src.shape[0]
        int ch = src.shape[1]
        int h_src = src.shape[2]
        int w_src = src.shape[3]
        int h_dst = dst.shape[2]
        int w_dst = dst.shape[3]
        int i
        double *src_ptr = &src[0][0][0][0]
        double *dst_ptr = &dst[0][0][0][0]
        int *ind_ptr = &max_ind[0][0][0][0]
        int total_dst = num * ch * h_dst * w_dst

    for i in range(total_dst):
        src_ptr[ind_ptr[i]] += dst_ptr[i]
