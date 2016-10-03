cimport cython

cpdef int max_pooling_batch(float [:,:,:,:] src, float [:,:,:,:] dst, int [:,:,:,:] max_ind,
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
        int src_ind_actual = 0
        float *src_ptr = &src[0][0][0][0]
        float *dst_ptr = &dst[0][0][0][0]
        int *ind_ptr = &max_ind[0][0][0][0]
        float max_val
        int current_max_ind
        float tmp
        int max_h = h_src - win_h + 1
        int max_w = w_src - win_w + 1

    for i in range(num):
        for j in range(ch):
            for k in range(0, max_h, stride_h):
                for l in range(0, max_w, stride_w):
                    src_ind_actual = src_ind + k * w_src + l
                    max_val = -1e8
                    for m in range(win_h):
                        for n in range(win_w):
                            tmp = src_ptr[src_ind_actual + n]
                            if tmp > max_val:
                                max_val = tmp
                                current_max_ind = src_ind_actual + n
                        src_ind_actual += w_src
                    dst_ptr[dst_ind] = max_val
                    ind_ptr[dst_ind] = current_max_ind
                    dst_ind += 1
            src_ind += h_src * w_src
    return 0

cpdef int backprop_for_max_pooling(float [:,:,:,:] src, float [:,:,:,:] dst, int [:,:,:,:] max_ind):
    cdef:
        int num = src.shape[0]
        int ch = src.shape[1]
        int h_src = src.shape[2]
        int w_src = src.shape[3]
        int h_dst = dst.shape[2]
        int w_dst = dst.shape[3]
        int i
        float *src_ptr = &src[0][0][0][0]
        float *dst_ptr = &dst[0][0][0][0]
        int *ind_ptr = &max_ind[0][0][0][0]
        int total_dst = num * ch * h_dst * w_dst

    for i in range(total_dst):
        src_ptr[ind_ptr[i]] += dst_ptr[i]

@cython.cdivision(True)
cpdef int mean_pooling_batch(float [:,:,:,:] src, float [:,:,:,:] dst, int [:,:,:,:] mean_ind,
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
        int src_ind_actual = 0
        float *src_ptr = &src[0][0][0][0]
        float *dst_ptr = &dst[0][0][0][0]
        int *ind_ptr = &mean_ind[0][0][0][0]
        float window_total
        float tmp
        int mean_h = h_src - win_h + 1
        int mean_w = w_src - win_w + 1
        float win_size = win_h * win_w

    for i in range(num):
        for j in range(ch):
            for k in range(0, mean_h, stride_h):
                for l in range(0, mean_w, stride_w):
                    src_ind_actual = src_ind + k * w_src + l
                    window_total = 0.0
                    for m in range(win_h):
                        for n in range(win_w):
                            window_total += src_ptr[src_ind_actual + n]
                        src_ind_actual += w_src
                    dst_ptr[dst_ind] = window_total / win_size
                    ind_ptr[dst_ind] = src_ind + k * w_src + l
                    dst_ind += 1
            src_ind += h_src * w_src
    return 0

@cython.cdivision(True)
cpdef int backprop_for_mean_pooling(float [:,:,:,:] src, float [:,:,:,:] dst,
                                    int [:,:,:,:] mean_ind, int win_h, int win_w):
    cdef:
        int num = src.shape[0]
        int ch = src.shape[1]
        int h_src = src.shape[2]
        int w_src = src.shape[3]
        int h_dst = dst.shape[2]
        int w_dst = dst.shape[3]
        int i, j, k
        float *src_ptr = &src[0][0][0][0]
        float *dst_ptr = &dst[0][0][0][0]
        int *ind_ptr = &mean_ind[0][0][0][0]
        int total_dst = num * ch * h_dst * w_dst
        int start_pos
        float val
        float win_size = win_w * win_h

    for i in range(total_dst):
        start_pos = ind_ptr[i]
        val = dst_ptr[i] / win_size
        for j in range(win_h):
            for k in range(win_w):
                src_ptr[start_pos + j * w_src + k] += val
