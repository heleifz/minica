# 首先对图像实现 im2col 的算法
cimport cython
# 利用 scipy 自带的 blas 做矩阵运算
cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack
from cython cimport view

# 支持 3d array 的 im2col
@cython.boundscheck(False)
@cython.wraparound(False)
cdef int im2col_core(double [:,:,:] src, double [:,:] dst,
                     int kernel_h, int kernel_w) nogil:
    cdef:
        double *src_ptr = &src[0][0][0]
        double *dst_ptr = &dst[0][0]

        int c = src.shape[0]
        int h = src.shape[1]
        int w = src.shape[2]

        int result_h = h - kernel_h + 1
        int result_w = w - kernel_w + 1

        int result_total = result_h * result_w
        int filter_channel_total = kernel_h * kernel_w

    if dst.shape[0] != filter_channel_total * c or \
       dst.shape[1] != result_total:
        # dst array has different shape
        return -1

    # copy row by row
    cdef:
        int i, j, ch, n, r_index, img_offset
        double *current_row
        double *src_submatrix = src_ptr
    for ch in range(c):
        for i in range(kernel_h):
            for j in range(kernel_w):
                # 确定要拷贝的 image range
                # 确定目标的行
                r_index = ch * filter_channel_total + i * kernel_w + j
                current_row = dst_ptr + r_index * result_total
                src_submatrix = src_ptr + ch * w * h + i * w + j
                # 采用 dlacpy 稍微加速拷贝 (~10%)
                lapack.dlacpy('A', &result_w, &result_h,
                              src_submatrix, &h, current_row, &result_h)
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int im2col_n_to_one_core(double [:,:,:] src, double [:,:] dst,
                     int kernel_h, int kernel_w) nogil:
    cdef:
        double *src_ptr = &src[0][0][0]
        double *dst_ptr = &dst[0][0]

        int c = src.shape[0]
        int h = src.shape[1]
        int w = src.shape[2]

        int result_h = h - kernel_h + 1
        int result_w = w - kernel_w + 1

        int result_total = result_h * result_w
        int filter_channel_total = kernel_h * kernel_w

    if dst.shape[0] != filter_channel_total or \
       dst.shape[1] != result_total * c:
        # dst array has different shape
        return -1

    # copy row by row
    cdef:
        int i, j, ch, n, r_index, img_offset
        double *current_row
        double *src_submatrix = src_ptr

    for i in range(kernel_h):
        for j in range(kernel_w):
            # 确定要拷贝的 image range
            # 确定目标的行
            r_index = i * kernel_w + j
            current_row = dst_ptr + r_index * result_total * c
            src_submatrix = src_ptr + i * w + j
            for ch in range(c):
                # 采用 dlacpy 稍微加速拷贝 (~10%)
                lapack.dlacpy('A', &result_w, &result_h,
                              src_submatrix, &h, current_row, &result_h)
                src_submatrix += w * h
                current_row += result_w * result_h
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int conv_batch(double [:,:,:,:] src, double[:,:,:,:] dst,
                     double [:,:,:,:] kernel, double [:,:] buf) nogil:

    cdef int i
    cdef int num = src.shape[0]
    cdef int kn = kernel.shape[0]
    cdef int kh = kernel.shape[2]
    cdef int kw = kernel.shape[3]
    cdef int dst_single_size = dst.shape[1] * dst.shape[2] * dst.shape[3]

    cdef double *kernel_ptr = &kernel[0][0][0][0]
    cdef double *dst_ptr = &dst[0][0][0][0]
    cdef int h_buf, w_buf, h_vec, w_vec
    # fortran 时顺序是反的
    h_buf = buf.shape[1]
    w_buf = buf.shape[0]
    h_vec = w_buf
    w_vec = kn
    # 设置 stride
    cdef int ld_buf = h_buf
    cdef int ld_vec = h_vec
    cdef int ld_out = buf.shape[1]
    cdef double alpha = 1.0, beta = 0.0

    # 每个图片做一次乘法(其实一个乘法也行，但是内存膨胀比较厉害)
    for i in range(num):
        if im2col_core(src[i], buf, kh, kw) != 0:
            return -1
        # 调用 blas 乘法 routine
        blas.dgemm("N", "N", &h_buf, &w_vec, &w_buf, &alpha,
                   &buf[0][0], &ld_buf, kernel_ptr, &ld_vec,
                   &beta, dst_ptr, &ld_out)
        dst_ptr += dst_single_size
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void copy_and_pad(double [:,:,:] src, double [:,:,:] dst, int pad_h, int pad_w) nogil:

    cdef:
        double *src_ptr = &src[0][0][0]
        double *dst_ptr = &dst[0][pad_h][pad_w]
        int ch
        int c = src.shape[0]
        int h_src = src.shape[1], w_src = src.shape[2]
        int h_dst = dst.shape[1], w_dst = dst.shape[2]

    for ch in range(c):
        lapack.dlacpy('A', &w_src, &h_src,
                      src_ptr, &h_src, dst_ptr, &h_dst)
        dst_ptr += h_dst * w_dst
        src_ptr += h_src * w_src

    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void flip(double [:,:,:,:] kernel, double [:,:,:,:] kernel_t) nogil:
    # 一个个 channel copy
    cdef:
        double *src_ptr = &kernel[0][0][0][0]
        double *dst_ptr = &kernel_t[0][0][0][0]
        double *current_dst
        int num_src = kernel.shape[0]
        int c_src = kernel.shape[1]
        int num_dst = c_src
        int c_dst = num_src
        int h = kernel.shape[2]
        int w = kernel.shape[3]
        int i, j, k, l

    for i in range(num_src):
        for j in range(c_src):
            current_dst = dst_ptr + j * c_dst * h * w + i * h * w
            for k in range(h):
                for l in range(w):
                    current_dst[(h - k - 1) * w + (w - l - 1)] = src_ptr[k * w + l]
            src_ptr += h * w
    return

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int backward_for_conv_batch(double [:,:,:,:] src, double[:,:,:,:] dst,
                                  double [:,:,:,:] kernel,
                                  double [:,:,:] pad_buf,
                                  double [:,:] conv_buf):
    """
    对于 input 的反向传播
    tricky 的地方是必须外部提供两个分配好的 buffer，且 size 要计算好
    """
    cdef:
        int num = src.shape[0]

        int c_src = src.shape[1]
        int h_src = src.shape[2]
        int w_src = src.shape[3]

        int c_dst = dst.shape[1]
        int h_dst = dst.shape[2]
        int w_dst = dst.shape[3]

        int kh_t = kernel.shape[3]
        int kw_t = kernel.shape[2]

        int i, j, k, l

        double [:,:,:,:] kernel_t = \
            view.array(shape=(c_src, kernel.shape[0], kh_t, kw_t), itemsize=8, format="d")

        # fortran 时顺序是反的
        int h_buf = conv_buf.shape[1]
        int w_buf = conv_buf.shape[0]
        int h_vec = w_buf
        int w_vec = kernel.shape[1]
        # 设置 stride
        int ld_buf = h_buf
        int ld_vec = h_vec
        int ld_out = conv_buf.shape[1]
        double alpha = 1.0
        # beta 为 1 表示乘法结果是累加的，这要求 src 被 0 初始化
        double beta = 1.0
        double *kernel_ptr = &kernel_t[0][0][0][0]
        double *src_ptr = &src[0][0][0][0]

    with nogil:

        # 先转置 kernel
        flip(kernel, kernel_t)

        # 一个个图片处理
        for i in range(num):

            copy_and_pad(dst[i], pad_buf, kh_t - 1, kw_t - 1)
            if im2col_core(pad_buf, conv_buf, kh_t, kw_t) != 0:
                return -1
            # 调用 blas 乘法 routine
            blas.dgemm("N", "N", &h_buf, &w_vec, &w_buf, &alpha,
                       &conv_buf[0][0], &ld_buf, kernel_ptr, &ld_vec,
                       &beta, src_ptr, &ld_out)

            src_ptr += c_src * h_src * w_src

    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int backward_kernel_for_conv_batch(double [:,:,:,:] src, double[:,:,:,:] dst,
                                         double [:,:,:,:] kernel, double [:,:] buf) nogil:
    """
    对于 kernel 的反向传播
    """
    cdef:
        int num = src.shape[0]
        int c_src = src.shape[1]
        int h_src = src.shape[2]
        int w_src = src.shape[3]

        int c_dst = dst.shape[1]
        int h_dst = dst.shape[2]
        int w_dst = dst.shape[3]

        int h_kernel = kernel.shape[2]
        int w_kernel = kernel.shape[3]

        int i, j, k, l

        # fortran 时顺序是反的
        int h_buf = buf.shape[1]
        int w_buf = buf.shape[0]
        int h_vec = w_buf

        int w_vec = c_dst
        # 设置 stride
        int ld_buf = h_buf
        int ld_vec = h_vec
        int ld_out = buf.shape[1]
        double alpha = 1.0
        # beta 为 1 表示乘法结果是累加的，这要求 kernel 梯度被 0 初始化
        double beta = 1.0

        double *kernel_ptr = &kernel[0][0][0][0]
        double *src_ptr = &src[0][0][0][0]
        double *dst_ptr = &dst[0][0][0][0]

    for i in range(num):

        if im2col_n_to_one_core(src[i], buf, h_dst, w_dst) != 0:
            return -1

        # 调用 blas 乘法 routine
        blas.dgemm("N", "N", &h_buf, &w_vec, &w_buf, &alpha,
                   &buf[0][0], &ld_buf, dst_ptr, &ld_vec,
                   &beta, kernel_ptr, &ld_out)

        dst_ptr += w_dst * h_dst * c_dst

    return 0
